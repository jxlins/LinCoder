"""Prompt 组装与上下文预算控制。

这个模块负责决定：每一轮到底把多少 prefix、memory、相关笔记、历史
以及当前用户请求送进模型。
"""

from __future__ import annotations

import json
from dataclasses import dataclass


DEFAULT_TOTAL_BUDGET = 12000
DEFAULT_SECTION_BUDGETS = {
    "prefix": 3600,
    "memory": 1600,
    "relevant_memory": 1200,    # 相关记忆/检索内容: 1200
    "history": 5200,
}
DEFAULT_SECTION_FLOORS = {
    "prefix": 1200,
    "memory": 400,
    "relevant_memory": 300,
    "history": 1500,
}
# 当 prompt 超预算时，会优先压缩这些 section。
DEFAULT_REDUCTION_ORDER = ("relevant_memory", "history", "memory", "prefix")    # 压缩优先级
SECTION_ORDER = ("prefix", "memory", "relevant_memory", "history", "current_request")   # 区块拼接顺序
CURRENT_REQUEST_SECTION = "current_request" # 当前用户请求这一块永远不裁剪
RELEVANT_MEMORY_LIMIT = 3   # 最多选 3 条相关记忆/检索内容进入 prompt

def _tail_clip(text, limit):
    """用于把文本限制在指定长度内"""
    text = str(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


@dataclass
class SectionRender:
    """
    这个类表示某个 prompt section 的渲染结果。
        raw：原始文本
        budget：分配给这一段的字符预算
        rendered：裁剪或压缩后的最终文本
        details：额外细节，例如渲染了哪些 note、压缩了多少历史
    """
    raw: str
    budget: int
    rendered: str
    details: dict | None = None

    @property
    def raw_chars(self):
        return len(self.raw)

    @property
    def rendered_chars(self):
        return len(self.rendered)


class ContextManager:
    """

        agent：当前 Lincoder agent 对象
        total_budget：整份 prompt 的总字符预算
        section_budgets：每个 section 的默认预算
        section_floors：每个 section 最低保留预算
        reduction_order：超预算时的裁剪顺序
    """
    def __init__(
        self,
        agent,
        total_budget=DEFAULT_TOTAL_BUDGET,
        section_budgets=None,
        section_floors=None,
        reduction_order=None,
    ):
        self.agent = agent
        self.total_budget = int(total_budget)
        self.section_budgets = dict(DEFAULT_SECTION_BUDGETS)
        if section_budgets:
            self.section_budgets.update({str(key): int(value) for key, value in section_budgets.items()})
        self._section_floor_overrides = {str(key): int(value) for key, value in (section_floors or {}).items()}
        self.section_floors = self._compute_section_floors()
        self.reduction_order = tuple(reduction_order or DEFAULT_REDUCTION_ORDER)

    def build(self, user_message):
        """按预算组装一轮完整 prompt。

        为什么存在：
        仅靠用户这一轮输入，模型并不知道当前仓库状态、会话里已经读过什么、
        哪些旧信息还值得继续参考。这个函数负责把“稳定基线 + 工作记忆 +
        相关笔记 + 历史 + 当前请求”拼成真正发给模型的 prompt。

        输入 / 输出：
        - 输入：`user_message`，也就是用户当前这一轮的新请求。
        - 输出：`(prompt, metadata)`。
          `prompt` 是最终发送给模型的文本；
          `metadata` 记录了每个 section 的原始长度、裁剪后的长度、是否触发了
          预算收缩等信息，后续会进入 trace/report，便于解释这轮 prompt
          是怎么被拼出来的。

        在 agent 链路里的位置：
        它位于 `Lincoder.ask()` 的每轮模型调用之前，是“真正发请求给模型”
        的最后一道组装工序。`WorkspaceContext` 提供稳定前缀，`LayeredMemory`
        提供工作记忆，这个函数则把它们和当前请求合成一份可控大小的 prompt。
        """
        user_message = str(user_message)
        self.section_floors = self._compute_section_floors()

        """
        1. 读取 feature flags。
            从 agent 中读取三个功能开关
            memory：是否启用普通记忆
            relevant_memory：是否启用相关记忆召回
            context_reduction：是否启用上下文压缩
        """
        memory_enabled = True
        relevant_memory_enabled = True
        context_reduction_enabled = True

        if hasattr(self.agent, "feature_enabled"):
            memory_enabled = self.agent.feature_enabled("memory")
            relevant_memory_enabled = self.agent.feature_enabled("relevant_memory")
            context_reduction_enabled = self.agent.feature_enabled("context_reduction")

        """
        2. 构造 section 原始文本
            prefix 来自 build_prompt_prefix，一般包含：
                agent 身份
                工具调用规则
                工具列表
                工作区摘要
                有效输出格式示例
            memory 来自 agent.memory_text()，
                包含 LayeredMemory 渲染出来的任务摘要、最近文件、文件摘要、情景笔记
            history 渲染由 _render_history_section 单独完成，history 需要复杂压缩逻辑。
            当前用户请求单独作为一个 section
        """
        section_texts = {
            "prefix": str(getattr(self.agent, "prefix", "")),
            "memory": "Memory:\n- disabled" if not memory_enabled else str(self.agent.memory_text()),
            "history": "",
            CURRENT_REQUEST_SECTION: f"Current user request:\n{user_message}",
        }

        """
        3. 加入 checkpoint 文本：如果 agent 提供了 checkpoint 渲染能力，就把 checkpoint 文本追加到 prefix 后面。
            checkpoint 文本通常包含：
                上次任务停在哪里
                当前恢复状态
                工作区是否发生变化
                下一步建议
                关键文件
        """
        checkpoint_text = ""
        if hasattr(self.agent, "render_checkpoint_text"):
            checkpoint_text = str(self.agent.render_checkpoint_text() or "").strip()
        if checkpoint_text:
            section_texts["prefix"] = section_texts["prefix"] + "\n\n" + checkpoint_text

        """
        4. 召回相关记忆
            如果记忆系统开启，并且支持相关记忆召回，就根据当前用户请求检索相关 notes。
        """
        selected_notes = []
        if memory_enabled and relevant_memory_enabled and hasattr(self.agent, "memory") and hasattr(self.agent.memory, "retrieval_candidates"):
            selected_notes = self.agent.memory.retrieval_candidates(user_message, limit=RELEVANT_MEMORY_LIMIT)

        """
        5. 如果 context_reduction 关闭，系统不按预算裁剪。它会：
            完整渲染 prefix
            完整渲染 memory
            完整渲染 relevant_memory
            完整渲染 history
            完整保留当前请求
        然后直接组装 prompt 和 metadata 返回。
        """
        if not context_reduction_enabled:
            rendered = self._render_sections_without_reduction(section_texts, selected_notes=selected_notes)
            prompt = self._assemble_prompt(rendered)
            metadata = self._metadata(
                prompt=prompt,
                rendered=rendered,
                budgets={section: render.budget for section, render in rendered.items() if section != CURRENT_REQUEST_SECTION},
                reduction_log=[],
                selected_notes=selected_notes,
                user_message=user_message,
                section_texts=section_texts,
            )
            return prompt, metadata

        """
        6. 启用上下文压缩时的整体流程
            复制每个 section 的默认预算
            按预算渲染各 section
            组装成完整 prompt
            准备记录压缩日志
        """
        budgets = dict(self.section_budgets)
        rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
        prompt = self._assemble_prompt(rendered)
        reduction_log = []

        # 如果 prompt 超预算，就按固定顺序不断压缩。
        # 这里的顺序体现了平台偏好：
        # 先牺牲 relevant_memory，再牺牲 history，然后才动 memory 和 prefix。
        # 最新用户请求永远不裁剪，因为那是本轮最重要的输入。
        while len(prompt) > self.total_budget:
            overflow = len(prompt) - self.total_budget
            reduced = False
            for section in self.reduction_order:
                floor = int(self.section_floors.get(section, 0))
                current_budget = int(budgets.get(section, 0))
                if current_budget <= floor:
                    continue
                new_budget = max(floor, current_budget - overflow)
                if new_budget >= current_budget:
                    continue
                reduction_log.append(
                    {
                        "section": section,
                        "before_chars": current_budget,
                        "after_chars": new_budget,
                        "overflow_chars": overflow,
                    }
                )
                budgets[section] = new_budget
                rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
                prompt = self._assemble_prompt(rendered)
                reduced = True
                break
            if not reduced:
                break

        metadata = self._metadata(
            prompt=prompt,
            rendered=rendered,
            budgets=budgets,
            reduction_log=reduction_log,
            selected_notes=selected_notes,
            user_message=user_message,
            section_texts=section_texts,
        )
        return prompt, metadata

    def _render_sections_without_reduction(self, section_texts, selected_notes=None):
        """
        无压缩渲染
        """
        selected_notes = selected_notes or []
        relevant_lines = ["Relevant memory:"]
        if selected_notes:
            relevant_lines.extend(f"- {note['text']}" for note in selected_notes)
        else:
            relevant_lines.append("- none")
        relevant_raw = "\n".join(relevant_lines)

        history = list(getattr(self.agent, "session", {}).get("history", []))
        history_raw = self._raw_history_text(history)

        return {
            "prefix": SectionRender(raw=section_texts["prefix"], budget=len(section_texts["prefix"]), rendered=section_texts["prefix"], details={}),
            "memory": SectionRender(raw=section_texts["memory"], budget=len(section_texts["memory"]), rendered=section_texts["memory"], details={}),
            "relevant_memory": SectionRender(
                raw=relevant_raw,
                budget=len(relevant_raw),
                rendered=relevant_raw,
                details={
                    "selected_notes": [note["text"] for note in selected_notes],
                    "rendered_notes": [note["text"] for note in selected_notes],
                    "selected_count": len(selected_notes),
                    "rendered_count": len(selected_notes),
                    "note_budget": 0,
                },
            ),
            "history": SectionRender(raw=history_raw, budget=len(history_raw), rendered=history_raw, details={"rendered_entries": []}),
            CURRENT_REQUEST_SECTION: SectionRender(
                raw=section_texts[CURRENT_REQUEST_SECTION],
                budget=0,
                rendered=section_texts[CURRENT_REQUEST_SECTION],
                details={},
            ),
        }

    def _compute_section_floors(self):
        """
        计算最低预算
        默认 floor 是每个 section 初始预算的四分之一，且至少 20 字符。
        """
        floors = {
            section: max(20, int(budget) // 4)
            for section, budget in self.section_budgets.items()
        }
        floors.update(self._section_floor_overrides)
        return floors

    def _render_sections(self, section_texts, budgets, selected_notes=None):
        """
        按预算渲染所有 section
        """
        rendered = {}
        for section in SECTION_ORDER:
            budget = budgets.get(section)

            # 当前请求不裁剪
            if section == CURRENT_REQUEST_SECTION:
                raw = section_texts[section]
                rendered[section] = SectionRender(raw=raw, budget=0, rendered=raw, details={})

            # 相关记忆需要特殊处理：每条 note 平分预算，避免某条 note 过长挤掉其他 note。
            elif section == "relevant_memory":
                rendered[section] = self._render_relevant_memory(selected_notes or [], int(budget or 0))

            # 历史记录有更复杂压缩策略
            elif section == "history":
                rendered[section] = self._render_history_section(int(budget or 0))

            # prefix 和 memory 直接用 _tail_clip 按预算裁剪
            else:
                raw = section_texts[section]
                rendered_text = _tail_clip(raw, int(budget)) if budget is not None else raw
                rendered[section] = SectionRender(raw=raw, budget=int(budget) if budget is not None else 0, rendered=rendered_text, details={})
        return rendered

    def _render_relevant_memory(self, selected_notes, budget):
        """
        渲染相关记忆
        """
        header = "Relevant memory:"
        note_texts = [str(note.get("text", "")) for note in selected_notes if str(note.get("text", "")).strip()]
        raw_lines = [header] + [f"- {text}" for text in note_texts]
        raw = "\n".join(raw_lines) if note_texts else "\n".join([header, "- none"])

        if not note_texts:
            rendered = raw
            return SectionRender(
                raw=raw,
                budget=budget,
                rendered=rendered,
                details={
                    "selected_notes": [],
                    "rendered_notes": [],
                    "selected_count": 0,
                    "rendered_count": 0,
                    "note_budget": 0,
                },
            )

        # 相关记忆按条平均分配预算
        per_note_budget = self._per_note_budget(budget, len(note_texts), header)
        rendered_notes = []
        while True:
            # 让每条 note 平分这一段的预算，避免一条超长笔记把其他笔记都挤掉。
            rendered_notes = [_tail_clip(text, per_note_budget) for text in note_texts]
            rendered = "\n".join([header] + [f"- {text}" for text in rendered_notes])
            if len(rendered) <= budget or per_note_budget <= 1:
                break
            per_note_budget -= 1

        if len(rendered) > budget and budget > 0:
            rendered = _tail_clip(raw, budget)
            rendered_notes = [rendered]

        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details={
                "selected_notes": note_texts,
                "rendered_notes": rendered_notes,
                "selected_count": len(note_texts),
                "rendered_count": len(rendered_notes),
                "note_budget": per_note_budget,
            },
        )

    def _per_note_budget(self, budget, note_count, header):
        """计算每条 note 的预算"""
        if note_count <= 0:
            return 0
        overhead = len(header) + 3 * note_count
        usable = max(0, budget - overhead)
        return max(1, usable // note_count)

    def _render_history_section(self, budget):
        """渲染历史记录"""

        # 1. 从 agent session 中取出历史
        history = list(getattr(self.agent, "session", {}).get("history", []))
        raw = self._raw_history_text(history)

        # 1.1 如果没有历史，返回空的渲染结果
        if not history:
            rendered = "Transcript:\n- empty"
            return SectionRender(
                raw=raw,
                budget=budget,
                rendered=rendered,
                details={
                    "rendered_entries": [],
                    "older_entries_count": 0,
                    "collapsed_duplicate_reads": 0,
                    "reused_file_summary_count": 0,
                    "summarized_tool_count": 0,
                },
            )

        # 优先保留最 6 条近的历史，因为下一步决策通常最依赖刚刚发生的工具结果。
        recent_window = 6
        recent_start = max(0, len(history) - recent_window)

        # 2. 先压缩历史条目，再倒序装回 prompt
        history_entries, history_details = self._compressed_history_entries(history, recent_start)
        rendered_entries = []
        for entry in reversed(history_entries):
            recent = bool(entry.get("recent", False))

            # 尝试把当前 entry 加到已有 rendered_entries 前面。
            candidate_lines = list(entry.get("lines", []))
            candidate_entries = candidate_lines + rendered_entries
            candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
            # 如果加入后不超预算，就保留
            if len(candidate_rendered) <= budget:
                rendered_entries = candidate_entries
                continue

            # 如果这条是最近历史，但完整加入会超预算，系统不会立刻放弃，而是尝试裁剪这条记录。
            if recent:
                available = budget - len("Transcript:")
                if rendered_entries:
                    available -= sum(len(line) + 1 for line in rendered_entries)
                available = max(20, available - 1)
                candidate_lines = [_tail_clip(line, available) for line in candidate_lines]
                candidate_entries = candidate_lines + rendered_entries
                candidate_rendered = "\n".join(["Transcript:", *candidate_entries])
                if len(candidate_rendered) <= budget:
                    rendered_entries = candidate_entries

            else:
                # 如果是旧历史，策略更激进：每行只保留 20 字符。
                smaller_lines = [_tail_clip(line, 20) for line in candidate_lines]
                smaller_entries = smaller_lines + rendered_entries
                smaller_rendered = "\n".join(["Transcript:", *smaller_entries])
                if len(smaller_rendered) <= budget:
                    rendered_entries = smaller_entries
        rendered = "\n".join(["Transcript:", *rendered_entries])

        # 如果最终 rendered 仍然超预算，就直接对完整 raw history 进行裁剪。
        if len(rendered) > budget and budget > 0:
            rendered = _tail_clip(raw, budget)

        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details={
                "recent_window": recent_window,
                "recent_start": recent_start,
                "rendered_entries": rendered_entries,
                **history_details,
            },
        )

    def _compressed_history_entries(self, history, recent_start):
        """
        历史对话压缩
        """
        entries = []
        seen_older_reads = set()
        details = {
            "older_entries_count": 0,
            "collapsed_duplicate_reads": 0,
            "reused_file_summary_count": 0,
            "summarized_tool_count": 0,
        }

        for index, item in enumerate(history):

            # 区分新旧对话：
            # 近期对话 (recent = True)：保留较高保真度，单条消息最多渲染 900行。
            # 早期对话 (recent = False)：进入下方的压缩逻辑，默认单条消息限制在 60行。
            recent = index >= recent_start
            if recent:
                line_limit = 900
                entries.append(
                    {
                        "recent": True,
                        "lines": self._render_history_item(item, line_limit),
                    }
                )
                continue

            """
            非近期的历史
                策略一：合并重复的文件读取 (read_file)
                    如果工具调用是 read_file，会提取文件路径 (path)。
                    如果该路径之前已经被读取过（存在于 seen_older_reads 集合中），则直接跳过（折叠），并记录 collapsed_duplicate_reads。
                    如果是首次读取，则尝试获取该文件的摘要 (_reusable_file_summary)。如果有摘要，就将其替换为 "路径 -> 摘要" 的极简格式。
                策略二：其他工具调用的摘要化
                    如果不是 read_file 但是其他类型的工具调用（如搜索、执行命令等），则调用 _summarize_old_tool_item(item) 生成一行摘要，替代原本可能非常冗长的工具输出结果。
                策略三：普通文本消息的截断
                    如果是普通的用户或助手发言，则调用 _render_history_item 并严格限制在 60行 以内。
            """
            if item["role"] == "tool" and item["name"] == "read_file":
                path = str(item["args"].get("path", "")).strip()
                if path in seen_older_reads:
                    details["collapsed_duplicate_reads"] += 1
                    continue
                seen_older_reads.add(path)
                summary = self._reusable_file_summary(path)
                if summary:
                    entries.append({"recent": False, "lines": [f"{path} -> {summary}"]})
                    details["older_entries_count"] += 1
                    details["reused_file_summary_count"] += 1
                    continue

            if item["role"] == "tool":
                summary_line = self._summarize_old_tool_item(item)
                entries.append({"recent": False, "lines": [summary_line]})
                details["older_entries_count"] += 1
                details["summarized_tool_count"] += 1
                continue

            entries.append({"recent": False, "lines": self._render_history_item(item, 60)})

        return entries, details

    def _reusable_file_summary(self, path):
        """访问 agent.memory，把某个 path 对应的 file summary 取出来。"""
        memory = getattr(self.agent, "memory", None)
        if memory is None or not hasattr(memory, "to_dict"):
            return ""

        snapshot = memory.to_dict()
        summary = snapshot.get("file_summaries", {}).get(str(path), {})
        if not summary:
            return ""
        return str(summary.get("summary", "")).strip()

    def _summarize_old_tool_item(self, item):
        """
        旧工具摘要

        对 run_shell 做特殊处理。提取
            command 输出的前 3 个非空行 组合成一行摘要。
            例如：
                pytest -q -> FAILED tests/test_api.py | AssertionError | exit_code: 1

        对于其他工具，只保留渲染后的第一行。
        """
        if item["name"] == "run_shell":
            command = str(item["args"].get("command", "")).strip() or "shell"
            lines = [line.strip() for line in str(item.get("content", "")).splitlines() if line.strip()]
            summary = " | ".join(lines[:3]) if lines else "(empty)"
            return f"{command} -> {summary}"
        return self._render_history_item(item, 60)[0]

    def _raw_history_text(self, history):
        """
        这个函数生成未压缩的历史文本

            格式是：
                Transcript:
                [user] ...
                [assistant] ...
                [tool:read_file] {"path": "main.py"}
                工具内容
        """
        if not history:
            return "Transcript:\n- empty"
        lines = []
        for item in history:
            if item["role"] == "tool":
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(str(item["content"]))
            else:
                lines.append(f"[{item['role']}] {item['content']}")
        return "\n".join(["Transcript:", *lines])

    def _render_history_item(self, item, line_limit):
        """
        单条历史渲染

            工具历史会有两行：
                [tool:read_file] {"path": "main.py"}
                文件内容或裁剪后的内容

            普通消息：
                return [f"[{item['role']}] {_tail_clip(item['content'], line_limit)}"]
        """
        if item["role"] == "tool":
            prefix = f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}"
            content = _tail_clip(item["content"], max(20, line_limit))
            return [prefix, content]
        return [f"[{item['role']}] {_tail_clip(item['content'], line_limit)}"]

    def _assemble_prompt(self, rendered):
        """
        组装最终 prompt

        按固定顺序拼接所有 section。顺序对应了 agent 的认知结构：
            先看规则和工具
            再看工作记忆
            再看与当前任务相关的记忆
            再看历史过程
            最后看当前请求

        当前请求放最后，可以强化模型对最新任务的关注。
        """
        # 顺序是刻意设计的：稳定规则放前面，最新请求放最后。
        return "\n\n".join(
            [
                rendered["prefix"].rendered,
                rendered["memory"].rendered,
                rendered["relevant_memory"].rendered,
                rendered["history"].rendered,
                rendered[CURRENT_REQUEST_SECTION].rendered,
            ]
        ).strip()

    def _metadata(self, prompt, rendered, budgets, reduction_log, selected_notes, user_message, section_texts):
        """生成 prompt 构造报告"""
        section_metadata = {}
        for section in SECTION_ORDER[:-1]:
            section_metadata[section] = {
                "raw_chars": rendered[section].raw_chars,
                "budget_chars": int(budgets.get(section, 0)),
                "rendered_chars": rendered[section].rendered_chars,
            }
        section_metadata[CURRENT_REQUEST_SECTION] = {
            "raw_chars": len(section_texts[CURRENT_REQUEST_SECTION]),
            "budget_chars": None,
            "rendered_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
        }
        return {
            "prompt_chars": len(prompt),
            "prompt_budget_chars": self.total_budget,
            "prompt_over_budget": len(prompt) > self.total_budget,
            "section_order": list(SECTION_ORDER),
            "section_budgets": {
                section: (None if section == CURRENT_REQUEST_SECTION else int(budgets.get(section, 0)))
                for section in SECTION_ORDER
            },
            "sections": section_metadata,
            "budget_reductions": reduction_log,
            "reduction_order": list(self.reduction_order),
            "relevant_memory": {
                "limit": RELEVANT_MEMORY_LIMIT,
                "selected_count": len(selected_notes),
                "selected_notes": [note["text"] for note in selected_notes],
                "selected_sources": [str(note.get("source", "")).strip() for note in selected_notes],
                "selected_kinds": [str(note.get("kind", "episodic")).strip() or "episodic" for note in selected_notes],
                "selected_durable_count": sum(
                    1 for note in selected_notes if (str(note.get("kind", "episodic")).strip() or "episodic") == "durable"
                ),
                "raw_chars": rendered["relevant_memory"].raw_chars,
                "rendered_chars": rendered["relevant_memory"].rendered_chars,
                "rendered_notes": list(rendered["relevant_memory"].details.get("rendered_notes", [])),
                "rendered_count": int(rendered["relevant_memory"].details.get("rendered_count", 0)),
            },
            "history": {
                "raw_chars": rendered["history"].raw_chars,
                "rendered_chars": rendered["history"].rendered_chars,
                "older_entries_count": int(rendered["history"].details.get("older_entries_count", 0)),
                "collapsed_duplicate_reads": int(rendered["history"].details.get("collapsed_duplicate_reads", 0)),
                "reused_file_summary_count": int(rendered["history"].details.get("reused_file_summary_count", 0)),
                "summarized_tool_count": int(rendered["history"].details.get("summarized_tool_count", 0)),
            },
            "current_request": {
                "text": user_message,
                "raw_chars": len(user_message),
                "rendered_chars": len(user_message),
                "section_chars": len(rendered[CURRENT_REQUEST_SECTION].rendered),
            },
        }