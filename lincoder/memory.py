"""
多步 agent 运行时使用的轻量工作记忆。

session history 负责保存完整事件流；这个模块只保存更小的一层工作集：
当前任务摘要、最近接触的文件、文件短摘要，以及少量跨轮笔记。
这样下一轮 prompt 还能接上上一轮，但不会被整段历史塞满。
"""

import hashlib
from datetime import datetime
import re
from pathlib import Path

WORKING_FILE_LIMIT = 8      # 最多记录 8 个最近相关文件
EPISODIC_NOTE_LIMIT = 12    # 最多记录 12 条短期过程笔记
FILE_SUMMARY_LIMIT = 6      # 渲染给模型时最多展示 6 个文件摘要

"""
长期记忆主题：
    project-conventions：项目约定（所有测试文件放在 tests 目录）
    key-decisions：关键决策（后端采用 FastAPI）
    dependency-facts：依赖和环境事实（项目使用 Python 3.10）
    user-preferences：用户偏好（项目使用 Python 3.10）
"""
DURABLE_TOPIC_DEFAULTS = {
    "project-conventions": {
        "title": "Project Conventions",
        "summary": "Stable repository conventions.",
        "tags": ["convention"],
    },
    "key-decisions": {
        "title": "Key Decisions",
        "summary": "Long-lived decisions and rationale anchors.",
        "tags": ["decision"],
    },
    "dependency-facts": {
        "title": "Dependency Facts",
        "summary": "Stable dependency and environment facts.",
        "tags": ["dependency"],
    },
    "user-preferences": {
        "title": "User Preferences",
        "summary": "Stable user preferences.",
        "tags": ["preference"],
    },
}

def default_memory_state():
    """
    一个默认的 memory state
    包含三类核心记忆：
        working：当前工作记忆
        episodic_notes：短期过程笔记
        file_summaries：文件摘要
    """

    return {
        "working": {
            "task_summary": "",
            "recent_files": [],
        },
        "episodic_notes": [],
        "file_summaries": {},
        "task": "",
        "files": [],
        "notes": [],
        "next_note_index": 0,
    }

class DurableMemoryStort:
    """
    长期记忆存储
    这个类负责把长期记忆写入磁盘
    """
    def __init__(self, root):
        self.root = Path(root)
        self.index_path = self.root / "MEMORY.md"
        self.topics_dir = self.root / "topics"

    def topic_slugs(self):
        return [topic["topic"] for topic in self.load_index()]

    def load_index(self):
        """
        这个方法读取 MEMORY.md，解析其中有哪些 memory topic。

        识别这种格式：
            - [project-conventions](topics/project-conventions.md): Project Conventions
                - summary: Stable repository conventions.
                - tags: convention

        解析后得到结构化列表：
            {
                "topic": "project-conventions",
                "title": "Project Conventions",
                "summary": "...",
                "tags": [...]
            }
        """
        if not self.index_path.exists():
            return []

        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        topics = []
        current = None

        for raw in lines:
            line = raw.strip()
            match = re.match(r"- \[([^\]]+)\]\([^)]+\):\s*(.+)", line)
            if match:
                current = {
                    "topic": match.group(1).strip(),
                    "title": match.group(2).strip(),
                    "summary": "",
                    "tags": [],
                }
                topics.append(current)
                continue
            if current is None:
                continue
            summary_match = re.match(r"- summary:\s*(.+)", line)
            if summary_match:
                current["summary"] = summary_match.group(1).strip()
                continue
            tags_match = re.match(r"- tags:\s*(.+)", line)
            if tags_match:
                current["tags"] = [tag.strip() for tag in tags_match.group(1).split(",") if tag.strip()]
        return topics

    def load_topic_notes(self, topic):
        """
        这个方法读取某个主题文件中的 notes。

        例如读取：
            topics/key-decisions.md

        提取 ## Notes 后面的条目，并转换成统一格式：
            {
                "text": "...",
                "tags": tags,
                "source": topic,
                "created_at": updated_at,
                "kind": "durable",
            }
        """
        path = self.topics_dir / f"{topic}.md"
        if not path.exists():
            return []

        lines = path.read_text(encoding="utf-8").splitlines()
        notes = []
        capture = False
        updated_at = ""
        tags = []

        for raw in lines:
            line = raw.strip()
            if line.startswith("- tags:"):
                tags = [tag.strip() for tag in line.split(":", 1)[1].split(",") if tag.strip()]
            elif line.startswith("- updated_at:"):
                updated_at = line.split(":", 1)[1].strip()
            elif line == "## Notes":
                capture = True
            elif capture and line.startswith("- "):
                notes.append(
                    {
                        "text": line[2:].strip(),
                        "tags": tags,
                        "source": topic,
                        "created_at": updated_at or now(),
                        "kind": "durable",
                    }
                )
        return notes

     @staticmethod
    def _subject_key(text):
         """
         这个方法尝试从一条记忆中抽出主语。
         用于判断新旧记忆是否说的是同一个对象。

         支持英文和中文模式，例如：
            xxx is ...
            xxx are ...
            xxx uses ...
            xxx should ...
            xxx 是 ...
            xxx 使用 ...

         """
         text = str(text).strip()
         patterns = (
             r"^(.+?)\s+is\s+.+$",
             r"^(.+?)\s+are\s+.+$",
             r"^(.+?)\s+uses?\s+.+$",
             r"^(.+?)\s+should\s+.+$",
             r"^(.+?)是.+$",
             r"^(.+?)使用.+$",
         )

         for pattern in patterns:
             match = re.match(pattern, text, re.I)
             if match:
                subject = " ".join(_tokenize(match.group(1)))
                return subject or None
         return None

    def retrieval_candidates(self, query, limit=3):
        """
        这个方法从长期记忆中检索相关笔记。
        在 Agent 的工作记忆（Durable Notes）中，根据用户的查询（Query），通过“标签精确匹配”、“关键词重叠度”和“时间新鲜度”三个维度，快速召回最相关的知识笔记。
        """

        # 1. 查询预处理：将用户传入的原始查询字符串进行分词（Tokenize），转换为一个 Token 集合。
        query_tokens = _tokenize(query)
        ranked = []

        # 2. 遍历所有的主题（Topic）及其下的笔记。
        #   对于每一条笔记，提取其标签集合（note_tags），
        #   并将笔记正文、所属主题标题以及标签全部合并，
        #   构建出一个庞大的候选 Token 集合（note_tokens）。
        for topic in self.load_index():
            notes = self.load_topic_notes(topic["topic"])
            for note in notes:
                note_tags = {tag.lower() for tag in note.get("tags", [])}
                note_tokens = _tokenize(note.get("text", "")) | _tokenize(topic.get("title", "")) | note_tags

                # 计算查询词与笔记标签的交集。如果有交集则为 1，否则为 0。
                exact_tag_match = int(bool(query_tokens & note_tags))
                #计算查询词与笔记全量 Token 集合的交集长度（即命中了多少个词）。
                keyword_overlap = len(query_tokens & note_tokens)
                if exact_tag_match == 0 and keyword_overlap == 0:
                    continue
                # 解析笔记的创建时间戳。
                recency = _parse_timestamp(note.get("created_at"))
                ranked.append(((exact_tag_match, keyword_overlap, recency), note))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [note for _, note in ranked[:limit]]

    def _write_index(self, topics):
        """
        索引文件生成器
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)

        lines = ["# Durable Memory Index", ""]

        for topic in topics:
            lines.append(f"- [{topic['topic']}](topics/{topic['topic']}.md): {topic['title']}")
            lines.append(f"  - summary: {topic['summary']}")
            lines.append(f"  - tags: {', '.join(topic['tags'])}")

        self.index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _write_topic(self, topic, notes):
        """
        主题笔记文件生成器
        """
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        meta = DURABLE_TOPIC_DEFAULTS[topic]
        lines = [
            f"# {meta['title']}",
            "",
            f"- topic: {topic}",
            f"- summary: {meta['summary']}",
            f"- tags: {', '.join(meta['tags'])}",
            f"- updated_at: {now()}",
            "",
            "## Notes",
        ]
        for note in notes:
            lines.append(f"- {note}")
        (self.topics_dir / f"{topic}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def promote(self, promotions):
        """
        长期记忆晋升与去重机制
        将 Agent 在运行过程中产生的高价值临时笔记（Working Memory），正式“晋升（Promote）”为持久化的长期知识（Durable Memory），同时自动处理知识的更新与覆盖。

        如果主题不存在，就创建主题
        如果同一条记忆已存在，就跳过
        如果新记忆和旧记忆主语相同，就替换旧记忆

        :return
            results：成功写入的记忆
            superseded：被新记忆替换的旧记忆
        """

        if not promotions:
            return [], []

        # 1. 加载现有知识库
        topics = {topic["topic"]: topic for topic in self.load_index()}
        topic_notes = {slug: [note["text"] for note in self.load_topic_notes(slug)] for slug in topics}
        results = []
        superseded = []

        # 2. 主题元数据初始化
        #   遍历待晋升的笔记。如果某个笔记属于一个全新的主题（即 topics 字典中还不存在该主题），
        #   则从默认配置（DURABLE_TOPIC_DEFAULTS）中提取元数据并初始化该主题。
        #   同时，确保该主题下的笔记列表已存在。
        for topic, note_text in promotions:
            meta = DURABLE_TOPIC_DEFAULTS[topic]
            topics.setdefault(
                topic,
                {
                    "topic": topic,
                    "title": meta["title"],
                    "summary": meta["summary"],
                    "tags": list(meta["tags"]),
                },
            )
            existing = topic_notes.setdefault(topic, [])

            # 3. 核心去重与覆盖逻辑
            if note_text in existing:
                continue

            new_subject = self._subject_key(note_text)
            replaced = False
            if new_subject:
                for index, old_text in enumerate(list(existing)):
                    # 如果新记忆和旧记忆主语相同，就替换旧记忆
                    if self._subject_key(old_text) == new_subject:
                        superseded.append(f"{topic}: {old_text} -> {note_text}")
                        existing[index] = note_text
                        replaced = True
                        break
            if not replaced:
                existing.append(note_text)
            results.append(f"{topic}: {note_text}")

        self._write_index([topics[slug] for slug in sorted(topics)])
        for topic, notes in topic_notes.items():
            self._write_topic(topic, notes)
        return results, superseded

def _ensure_list(value):
    """
    统一数据，将其他类型的数据转换为 List
    """

    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _dedupe_preserve_order(items):
    """保序去重"""
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result

def resolve_workspace_path(raw_path, workspace_root=None):
    """
    安全的工作区路径解析器（Safe Workspace Path Resolver）。
    将用户或 AI 传入的原始路径字符串，安全地转换为绝对物理路径，同时通过严格的边界校验，确保该路径绝对不会逃逸出指定的工作区（Workspace Root）。

    把路径解析到工作区内部。
    如果路径逃出工作区，就返回 None。
    """

    path = Path(str(raw_path))
    if workspace_root is None:
        return path

    # 调用 resolve() 解析工作区根目录，消除其中的符号链接和相对路径符号。
    root = Path(workspace_root).resolve()
    candidate = path if path.is_absolute() else root / path

    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved

def canonicalize_path(raw_path, workspace_root=None):
    """
    把路径标准化成相对工作区路径。

    例如：
        /Users/a/project/src/main.py

    会变成：
        src/main.py
    """

    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None:
        return Path(str(raw_path)).as_posix()
    if workspace_root is None:
        return Path(str(raw_path)).as_posix()
    root = Path(workspace_root).resolve()
    return resolved.relative_to(root).as_posix()


def file_freshness(raw_path, workspace_root=None):
    """
    计算文件内容的 SHA256。
    可以用它判断某个文件摘要是否过期。
    """

    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()

def _tokenize(text):
    """将一段自然语言文本转化为一个去重、归一化、无标点的 Token 集合。"""
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", str(text))}

def _parse_timestamp(value):
    """字符串转时间戳，失败时返回 0"""
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:
        return 0.0

def _normalize_note(note, index):
    """
    把一条笔记规范化成统一格式。
    作用：无论上游传入的笔记数据是简单的字符串、非字典对象，还是结构复杂的字典，它都能将其“清洗”并转换为系统内部统一的、类型安全的标准格式。
    """

    # 1. 纯文本字符串的快捷处理
    if isinstance(note, str):
        text = clip(note.strip(), 500)
        return {
            "text": text,
            "tags": [],
            "source": "",
            "created_at": now(),
            "note_index": index,
            "kind": "episodic",
        }

    # 2. 非字典对象的兜底转换
    if not isinstance(note, dict):
        text = clip(str(note).strip(), 500)
        return {
            "text": text,
            "tags": [],
            "source": "",
            "created_at": now(),
            "note_index": index,
            "kind": "episodic",
        }

    # 3. 标准字典的深度清洗与类型转换
    text = clip(str(note.get("text", "")).strip(), 500)
    tags = [str(tag).strip() for tag in _ensure_list(note.get("tags", [])) if str(tag).strip()]
    source = str(note.get("source", "")).strip()
    created_at = str(note.get("created_at", "")).strip() or now()
    note_index = int(note.get("note_index", index))
    kind = str(note.get("kind", "episodic")).strip() or "episodic"

    return {
        "text": text,
        "tags": _dedupe_preserve_order(tags),
        "source": source,
        "created_at": created_at,
        "note_index": note_index,
        "kind": kind,
    }

def normalize_memory_state(state, workspace_root=None):
    """
    是把任意来源的 memory state 统一整理成当前 runtime 可以直接使用的结构。

    原因：memory state 可能来自：
        新会话
        旧版本 session 文件
        手动编辑过的 JSON
        上一次运行中断时保存的状态
    这些状态的字段可能不完整、类型不规范、路径不标准。所以这个函数会统一处理。
    """

    if state is None:
        state = default_memory_state()
    elif not isinstance(state, dict):
        raise TypeError("memory state must be a mapping")

    # 1. 规范化 working memory
    # 规范化层的作用，是把“磁盘里可能长得不太一样的旧状态”
    # 统一整理成当前 runtime 可直接使用的紧凑结构。
    working = state.get("working")
    if not isinstance(working, dict):
        working = {}
    working.setdefault("task_summary", "")
    working.setdefault("recent_files", [])
    working["task_summary"] = clip(str(working.get("task_summary", "")).strip(), 300)
    # 对 recent_files 进行列表化（_ensure_list）、路径安全解析（canonicalize_path）、保序去重，最后截取最近的 WORKING_FILE_LIMIT 个文件。
    working["recent_files"] = _dedupe_preserve_order(
        [
            canonicalize_path(path, workspace_root)
            for path in _ensure_list(working.get("recent_files", []))
            if str(path).strip()
        ]
    )[-WORKING_FILE_LIMIT:]
    state["working"] = working

    # 2. 兼容旧字段
    #   如果旧状态中有 task 字段，但新结构中的 working.task_summary 为空，就把旧字段迁移过来。
    if not str(working["task_summary"]).strip() and state.get("task"):
        working["task_summary"] = clip(str(state.get("task", "")).strip(), 300)
    #   会把旧的 files 迁移到 working.recent_files。
    if not working["recent_files"] and state.get("files"):
        working["recent_files"] = _dedupe_preserve_order(
            [
                canonicalize_path(path, workspace_root)
                for path in _ensure_list(state.get("files", []))
                if str(path).strip()
            ]
        )[-WORKING_FILE_LIMIT:]

    # 3. 规范化 episodic notes：利用 _normalize_note 对每一条笔记进行类型转换、去重和截断。最后，只保留最近的 EPISODIC_NOTE_LIMIT 条笔记。
    episodic_notes = state.get("episodic_notes")
    if not isinstance(episodic_notes, list):
        episodic_notes = []

    if not episodic_notes and state.get("notes"):
        episodic_notes = [
            _normalize_note(note, index)
            for index, note in enumerate(_ensure_list(state.get("notes", [])))
            if str(note).strip()
        ]
    else:
        normalized_notes = []
        for index, note in enumerate(episodic_notes):
            if isinstance(note, str) and not str(note).strip():
                continue
            normalized_notes.append(_normalize_note(note, index))
        episodic_notes = normalized_notes
    episodic_notes = episodic_notes[-EPISODIC_NOTE_LIMIT:]
    state["episodic_notes"] = episodic_notes

    # 4. 规范化 file summaries
    #   遍历文件摘要字典。对路径进行安全解析，对摘要内容进行截断。如果摘要格式不规范（纯字符串），则自动补齐时间戳和新鲜度（freshness）字段。过滤掉无效路径或空摘要。
    file_summaries = state.get("file_summaries")
    if not isinstance(file_summaries, dict):
        file_summaries = {}
    normalized_file_summaries = {}
    for path, summary in file_summaries.items():
        path = canonicalize_path(path, workspace_root)
        if isinstance(summary, dict):
            text = clip(str(summary.get("summary", "")).strip(), 500)
            created_at = str(summary.get("created_at", "")).strip() or now()
            freshness = summary.get("freshness")
            freshness = None if freshness in (None, "") else str(freshness).strip() or None
        else:
            text = clip(str(summary).strip(), 500)
            created_at = now()
            freshness = None
        if not path or not text:
            continue
        normalized_file_summaries[path] = {
            "summary": text,
            "created_at": created_at,
            "freshness": freshness,
        }
    state["file_summaries"] = normalized_file_summaries

    # 5. 笔记索引（Index）的安全修复
    next_note_index = state.get("next_note_index")
    if not isinstance(next_note_index, int) or next_note_index < 0:
        next_note_index = 0
    max_index = max([note["note_index"] for note in episodic_notes], default=-1)
    state["next_note_index"] = max(next_note_index, max_index + 1)

    # 6. 同步兼容字段
    state["task"] = working["task_summary"]
    state["files"] = list(working["recent_files"])
    state["notes"] = [note["text"] for note in episodic_notes]
    durable_root = Path(workspace_root) / ".lincoder" / "memory" if workspace_root is not None else None
    durable_store = DurableMemoryStore(durable_root) if durable_root is not None else None
    state["durable_topics"] = durable_store.topic_slugs() if durable_store is not None else []
    return state


def set_task_summary(state, summary, workspace_root=None):
    """
    把当前任务摘要写入工作记忆。
    """
    state = normalize_memory_state(state, workspace_root)
    state["working"]["task_summary"] = clip(str(summary).strip(), 300)
    state["task"] = state["working"]["task_summary"]
    return state


def remember_file(state, path, workspace_root=None):
    """
    当 Agent 反复操作同一个文件时，该文件会被移到列表的最末端（代表“最新鲜”），而最久未操作的文件会在达到上限时被自动挤出列表。
    这确保了 Agent 的工作记忆中始终只保留最相关的近期文件上下文。
    """
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    files = [item for item in state["working"]["recent_files"] if item != path]
    files.append(path)
    state["working"]["recent_files"] = files[-WORKING_FILE_LIMIT:]
    state["files"] = list(state["working"]["recent_files"])
    return state


def append_note(state, text, tags=(), source="", created_at=None, workspace_root=None, kind="episodic"):
    """
    添加一条过程笔记
    将 Agent 产生的新临时笔记安全地注入到内存状态中，同时自动处理标签清洗、全局 ID 自增、文本去重以及容量控制。
    """

    # 1. 状态归一化与文本清洗
    state = normalize_memory_state(state, workspace_root)
    text = clip(str(text).strip(), 500)
    if not text:
        return state

    # 2. 标签清洗与笔记对象组装
    normalized_tags = _dedupe_preserve_order(
        [str(tag).strip() for tag in _ensure_list(tags) if str(tag).strip()]
    )
    note = {
        "text": text,
        "tags": normalized_tags,
        "source": str(source).strip(),
        "created_at": str(created_at).strip() if created_at else now(),
        "note_index": int(state.get("next_note_index", 0)),
        "kind": str(kind).strip() or "episodic",
    }

    # 3. 全局索引自增
    state["next_note_index"] = note["note_index"] + 1

    # 4. 文本级去重与滑动窗口截断
    notes = [item for item in state["episodic_notes"] if item["text"] != note["text"]]
    notes.append(note)
    state["episodic_notes"] = notes[-EPISODIC_NOTE_LIMIT:]
    state["notes"] = [item["text"] for item in state["episodic_notes"]]
    return state

def set_file_summary(state, path, summary, workspace_root=None):
    """
    保存某个文件的短摘要。
    同时保存文件当前的新鲜度
    """
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    summary = clip(str(summary).strip(), 500)
    if not path or not summary:
        return state
    state["file_summaries"][path] = {
        "summary": summary,
        "created_at": now(),
        "freshness": file_freshness(path, workspace_root),
    }
    return state

def invalidate_file_summary(state, path, workspace_root=None):
    """
    删除某个文件摘要。
    """
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    if not path:
        return state
    state["file_summaries"].pop(path, None)
    return state

def invalidate_stale_file_summaries(state, workspace_root=None):
    """
    检查所有文件摘要是否过期
    """
    state = normalize_memory_state(state, workspace_root)
    invalidated = []
    for path, summary in list(state["file_summaries"].items()):
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("freshness") == current_freshness:
            continue
        invalidated.append(path)
        state["file_summaries"].pop(path, None)
    return state, invalidated

def summarize_read_result(result, limit=180):
    """
    把 read_file 的结果压缩成短摘要。

    我们不会把完整文件内容塞进记忆层，
    这里只保留足够提醒下一轮“刚刚读到了什么”的短摘要。

    处理逻辑是：
        去掉空行
        如果第一行是 markdown 标题，就跳过
        取前 3 行
        用竖线连接
        裁剪到 180 字符
    """
    lines = [line.strip() for line in str(result).splitlines() if line.strip()]
    if not lines:
        return "(empty)"
    if lines[0].startswith("# "):
        lines = lines[1:]
    if not lines:
        return "(empty)"
    summary = " | ".join(lines[:3])
    return clip(summary, limit)


def retrieval_candidates(state, query, limit=3, workspace_root=None):
    """
    从 短期过程笔记 和 长期记忆 中找出与 query 相关的记忆。
    先查 episodic_notes，再查 durable memory，最后把它们混在一起排序返回。

    排序依据是：
        tag 精确命中优先
        关键词重叠越多越靠前
        越新的笔记越靠前
        note_index 越大越靠前
    """

    state = normalize_memory_state(state, workspace_root)
    # 将查询语句分词
    query_tokens = _tokenize(query)
    ranked = []

    # 遍历情景记忆（episodic_notes）
    for note in state["episodic_notes"]:
        # 召回逻辑故意保持简单透明：先看 tag 精确命中，
        # 再看关键词重叠，最后看新旧程度。这里不引入 embedding。
        note_tags = {tag.lower() for tag in note.get("tags", [])}
        # 将每条笔记的文本、来源和标签合并为一个大的 Token 集合。
        note_tokens = _tokenize(note.get("text", "")) | _tokenize(note.get("source", "")) | note_tags
        exact_tag_match = int(bool(query_tokens & note_tags))
        keyword_overlap = len(query_tokens & note_tokens)

        # 计算两个核心指标：exact_tag_match（标签是否有交集，1 或 0）和 keyword_overlap（关键词重叠的数量）。如果两者都为 0，直接丢弃。
        if exact_tag_match == 0 and keyword_overlap == 0:
            continue

        recency = _parse_timestamp(note.get("created_at"))
        note_index = int(note.get("note_index", 0))
        # 提取时间戳和笔记索引，将 (指标元组, 笔记对象) 加入候选列表
        ranked.append(((exact_tag_match, keyword_overlap, recency, note_index), note))

    # 如果配置了工作区，则从磁盘上的 DurableMemoryStore 中加载长期记忆，并使用完全相同的打分逻辑将其混入候选列表。
    # 注意这里将 note_index 设为 -1，作为长期记忆的标识。
    if workspace_root is not None:
        durable_store = DurableMemoryStore(Path(workspace_root) / ".lincoder" / "memory")
        for note in durable_store.retrieval_candidates(query, limit=limit):
            note_tags = {tag.lower() for tag in note.get("tags", [])}
            note_tokens = _tokenize(note.get("text", "")) | _tokenize(note.get("source", "")) | note_tags
            exact_tag_match = int(bool(query_tokens & note_tags))
            keyword_overlap = len(query_tokens & note_tokens)
            recency = _parse_timestamp(note.get("created_at"))
            ranked.append(((exact_tag_match, keyword_overlap, recency, -1), note))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [note for _, note in ranked[:limit]]


def retrieval_view(state, query, limit=3, workspace_root=None):
    """把检索结果渲染成文本"""
    candidates = retrieval_candidates(state, query, limit=limit, workspace_root=workspace_root)
    lines = ["Relevant memory:"]

    if not candidates:
        lines.append("- none")
        return "\n".join(lines)

    for note in candidates:
        lines.append(f"- {note['text']}")
    return "\n".join(lines)


def render_memory_text(state, workspace_root=None):
    """
    这个函数生成给模型看的 memory 摘要。

    Memory:
    - task: 帮我修改 main.py
    - recent_files: main.py, tests/test_main.py
    - file_summaries:
        - main.py: ...
        - tests/test_main.py: ...
    - episodic_notes: 3
    - durable_topics: key-decisions, user-preferences
    """

    state = normalize_memory_state(state, workspace_root)
    # 这里渲染的是给模型看的紧凑“仪表盘”，不是完整回放。
    # 笔记正文默认不展开，只有在相关召回时才按需拿出来。
    lines = [
        "Memory:",
        f"- task: {state['working']['task_summary'] or '-'}",
        f"- recent_files: {', '.join(state['working']['recent_files']) or '-'}",
    ]

    # 遍历近期文件列表，但通过 [:FILE_SUMMARY_LIMIT] 严格限制最多只渲染几个文件的摘要
    summaries = []
    for path in state["working"]["recent_files"][:FILE_SUMMARY_LIMIT]:
        summary = state["file_summaries"].get(path, {})
        current_freshness = file_freshness(path, workspace_root)
        if summary.get("summary", "") and summary.get("freshness") == current_freshness:
            summaries.append(f"- {path}: {summary['summary']}")
    if summaries:
        lines.append("- file_summaries:")
        lines.extend(f"  {line}" for line in summaries)
    else:
        lines.append("- file_summaries: -")

    lines.append(f"- episodic_notes: {len(state['episodic_notes'])}")
    durable_topics = state.get("durable_topics", [])
    lines.append(f"- durable_topics: {', '.join(durable_topics) or '-'}")
    return "\n".join(lines)

def is_effectively_empty(state, workspace_root=None):
    """
    判断 memory 是否基本为空。
    如果没有任务摘要、没有最近文件、没有过程笔记、没有文件摘要，就返回 True
    """

    state = normalize_memory_state(state, workspace_root)
    return (
        not str(state["working"]["task_summary"]).strip()
        and not state["working"]["recent_files"]
        and not state["episodic_notes"]
        and not state["file_summaries"]
    )


class LayeredMemory:
    """
    面向外部的统一接口

    这个类是对前面所有函数的封装。
    让 LinCoder 不需要直接操作底层字典。
    """
    def __init__(self, state=None, workspace_root=None):
        self.workspace_root = workspace_root
        self.state = normalize_memory_state(state, workspace_root)
        self.durable_store = DurableMemoryStore(Path(workspace_root) / ".lincoder" / "memory") if workspace_root is not None else None

    def to_dict(self):
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return self.state

    def canonical_path(self, path):
        return canonicalize_path(path, self.workspace_root)

    def set_task_summary(self, summary):
        self.state = set_task_summary(self.state, summary, self.workspace_root)
        return self

    def remember_file(self, path):
        self.state = remember_file(self.state, path, self.workspace_root)
        return self

    def append_note(self, text, tags=(), source="", created_at=None, kind="episodic"):
        self.state = append_note(
            self.state,
            text,
            tags=tags,
            source=source,
            created_at=created_at,
            workspace_root=self.workspace_root,
            kind=kind,
        )
        return self

    def set_file_summary(self, path, summary):
        self.state = set_file_summary(self.state, path, summary, self.workspace_root)
        return self

    def invalidate_file_summary(self, path):
        self.state = invalidate_file_summary(self.state, path, self.workspace_root)
        return self

    def invalidate_stale_file_summaries(self):
        self.state, invalidated = invalidate_stale_file_summaries(self.state, self.workspace_root)
        return invalidated

    def retrieval_candidates(self, query, limit=3):
        return retrieval_candidates(self.state, query, limit=limit, workspace_root=self.workspace_root)

    def retrieval_view(self, query, limit=3):
        return retrieval_view(self.state, query, limit=limit, workspace_root=self.workspace_root)

    def render_memory_text(self):
        return render_memory_text(self.state, self.workspace_root)

    def promote_durable(self, promotions):
        """
        调用 DurableMemoryStore.promote，把短期候选内容沉淀到长期记忆文件。
        """
        
        if self.durable_store is None:
            return [], []

        self.state = normalize_memory_state(self.state, self.workspace_root)
        promoted, superseded = self.durable_store.promote(promotions)

        self.state = normalize_memory_state(self.state, self.workspace_root)
        return promoted, superseded