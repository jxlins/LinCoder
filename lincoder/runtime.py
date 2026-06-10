"""Agent 运行时核心逻辑。

Lincoder 就是包在模型外面的控制循环：负责组 prompt、解析模型输出、
校验并执行工具、写 trace、更新工作记忆，以及在合适的时候停下来。
"""

import json
import hashlib
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from .prompt_prefix import build_prompt_prefix
from . import tools as toolkit

# 默认的环境变量允许列表，Lincoder 只会将这些环境变量暴露给模型，防止泄露敏感信息
DEFAULT_SHELL_ENV_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "LOGNAME", "PATH", "PWD", "SHELL", "TERM", "TMPDIR", "TMP", "TEMP", "USER")
# 默认的功能开关设置，控制 Lincoder 的一些可选功能是否启用
DEFAULT_FEATURE_FLAGS = {
    "memory": True,
    "relevant_memory": True,
    "context_reduction": True,
    "prompt_cache": True,
}
# 用于检测模型输出中是否包含表示用户意图要将某些信息保存到持久记忆中的关键词的正则表达式模式，支持英文和中文的关键词
DURABLE_MEMORY_INTENT_PATTERN = re.compile(r"(?i)\b(capture|remember|save|store|persist|note)\b")
# 用于检测模型输出中是否包含表示用户意图要将某些信息保存到持久记忆中的关键词的正则表达式模式，支持中文的关键词
DURABLE_MEMORY_INTENT_ZH_PATTERN = re.compile(r"(记住|保存|记录|沉淀|长期记忆|持久记忆)")
# 用于从模型输出的文本中提取要保存到持久记忆中的信息的正则表达式模式列表，每个模式都关联一个记忆类型标签，支持英文和中文的格式
DURABLE_MEMORY_LINE_PATTERNS = (
    ("project-conventions", re.compile(r"(?i)^Project convention:\s*(.+)$")),
    ("key-decisions", re.compile(r"(?i)^Decision:\s*(.+)$")),
    ("dependency-facts", re.compile(r"(?i)^Dependency:\s*(.+)$")),
    ("user-preferences", re.compile(r"(?i)^Preference:\s*(.+)$")),
    ("project-conventions", re.compile(r"^项目约定：\s*(.+)$")),
    ("key-decisions", re.compile(r"^决策：\s*(.+)$")),
    ("dependency-facts", re.compile(r"^依赖：\s*(.+)$")),
    ("user-preferences", re.compile(r"^偏好：\s*(.+)$")),
)
# 用于检测模型输出中是否包含可能泄露敏感信息的关键词或模式的正则表达式模式，支持英文和中文的关键词，以及常见的 API 密钥格式
SECRET_SHAPED_TEXT_PATTERN = re.compile(r"(?i)(\b(api[_ -]?key|token|secret|password)\b|sk-[A-Za-z0-9_-]{6,})")

__all__ = ["Lincoder", "SessionStore"]

class Lincoder:
    def __init__(
            self,
            model_client,
            workspace,
            session_store,
            session=None,
            run_store=None,
            approval_policy="ask",
            max_steps=6,
            max_new_tokens=512,
            depth=0,
            max_depth=1,
            read_only=False,
            shell_env_allowlist=None,
            secret_env_names=None,
            feature_flags=None,
            allowed_tools=None
    ):
        self.model_client = model_client        # 保存模型客户端
        self.workspace = workspace              # 保存工作区对象
        self.root = Path(workspace.repo_root)   # 工作区根目录的 Path 对象
        self.session_store = session_store      # 会话存储对象，负责保存和加载会话数据
        self.approval_policy = approval_policy  # 工具执行审批策略，默认为 "ask"，即每次执行工具前都询问用户
        self.max_steps = max_steps              # 最大步骤数，控制 Pico 在停止前可以执行的最大工具调用次数
        self.max_new_tokens = max_new_tokens    # 模型生成的最大 token 数，控制每次模型调用可以生成的最大文本长度
        self.depth = depth                      # 当前递归深度，初始为 0
        self.max_depth = max_depth              # 最大递归深度，控制 Pico 在遇到需要递归调用自身的情况时允许的最大深度，超过后将停止并返回错误
        self.read_only = read_only              # 只读模式标志，默认为 False，设置为 True 后 Pico 将不会执行任何工具调用，只会生成文本输出，适用于调试和分析模型行为
        self.shell_env_allowlist = tuple(shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST)    # 允许暴露给模型的 shell 环境变量名称列表，默认为 DEFAULT_SHELL_ENV_ALLOWLIST 中定义的变量
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}    # 需要特殊处理的环境变量名称集合，Lincoder 将不会直接暴露这些变量的值给模型，而是提供一个占位符，模型可以通过工具调用来获取这些变量的值，适用于存储敏感信息的环境变量
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)    # 功能开关设置字典，控制 Lincoder 的一些可选功能是否启用，默认为 DEFAULT_FEATURE_FLAGS 中定义的设置，如果传入了 feature_flags 参数，则会覆盖默认设置中对应的项，允许用户自定义功能开关设置
        if feature_flags:
            self.feature_flags.update({str(key): bool(value) for key, value in feature_flags.items()})
        self.allowed_tools = self._normalize_allowed_tools(allowed_tools)   # 允许使用的工具名称集合，Lincoder 将只允许模型调用这些工具，如果 allowed_tools 参数为 None，则表示不限制工具调用
        self.run_store = run_store or RunStore(Path(workspace.repo_root) / ".lincoder" / "runs")    # 运行存储对象，负责保存和加载工具调用的运行数据，如果没有传入 run_store 参数，则默认使用工作区根目录下的 .lincoder/runs 目录作为存储位置

        # 保存对话历史、记忆、检查点，是这个 agent 的长期运行状态
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": memorylib.default_memory_state(),
        }
        self._ensure_session_shape()
        # 初始化记忆系统，使用 LayeredMemory 来管理不同层次的记忆状态，并将其保存在 session 中，以便在后续的工具调用和模型交互中使用和更新
        self.memory = memorylib.LayeredMemory(
            self.session.setdefault("memory", memorylib.default_memory_state()),
            workspace_root=self.root,
        )
        self.session["memory"] = self.memory.to_dict()

        self.tools = self._apply_tool_allowlist(self.build_tools())  # 构建工具列表，并应用工具允许列表进行过滤，确保只有被允许的工具可以被模型调用
        self.tool_executor = ToolExecutor(self)     # 创建工具执行器对象，负责执行模型调用的工具，并处理工具执行的结果和可能的异常
        self.prefix_state = self.build_prefix()     # 构建 prompt 前缀状态，包含一些固定的文本和动态的上下文信息，这些信息会在每次生成 prompt 时被用来构建最终的 prompt 文本
        self.prefix = self.prefix_state.text        # 保存当前的 prompt 前缀文本，初始为 prefix_state 中生成的文本，后续可能会根据上下文变化进行更新
        self.context_manager = ContextManager(self) # 创建上下文管理器对象，负责管理对话上下文的状态和变化，包括对话历史、记忆状态、检查点等信息，并提供接口供工具调用和模型交互使用
        self.resume_state = self.evaluate_resume_state()    # 评估当前的恢复状态，判断是否需要从之前的会话状态中恢复对话上下文，如果需要，则将恢复状态保存在 session 中，以便在后续的工具调用和模型交互中使用
        self.session_path = self.session_store.save(self.session) # 保存当前会话状态到会话存储中，并获取保存后的路径，方便后续的加载和更新
        self.current_task_state = None          # 当前任务状态，包含当前正在执行的工具调用的相关信息，如工具名称、输入参数、执行结果等，这些信息会在工具执行过程中被更新，并在生成 prompt 和评估模型输出时使用
        self.current_run_dir = None             # 当前运行目录，保存当前工具调用的运行数据的目录路径，这些数据包括工具输入、输出、执行日志等信息，方便后续的分析和调试
        self.last_prompt_metadata = {}          # 上一次生成 prompt 时的元数据信息，包含生成 prompt 时使用的工具调用信息、上下文状态、记忆状态等，这些信息会在生成 prompt 和评估模型输出时使用，帮助 Lincoder 理解模型的行为和决策过程
        self.last_completion_metadata = {}      # 上一次模型生成完成时的元数据信息，包含模型生成的文本内容、使用的工具调用信息、上下文状态、记忆状态等，这些信息会在评估模型输出时使用，帮助 Lincoder 理解模型的行为和决策过程，并进行相应的工具调用和上下文更新
        self.last_durable_promotions = []       # 上一次模型输出中被识别为具有持久记忆意图的信息列表，这些信息是从模型输出的文本中提取出来的，并且被认为是用户希望保存到持久记忆中的重要信息，Lincoder 会将这些信息保存到 memory 中，并在后续的工具调用和模型交互中使用这些信息来提供更丰富的上下文和更准确的响应
        self.last_durable_rejections = []       # 上一次模型输出中被识别为具有持久记忆意图但被拒绝保存的信息列表，这些信息是从模型输出的文本中提取出来的，并且被认为是用户希望保存到持久记忆中的重要信息，但由于某些原因（如敏感信息、格式不正确等）被 Lincoder 拒绝保存到 memory 中，Lincoder 会记录这些信息，并在后续的工具调用和模型交互中避免使用这些信息，以防止泄露敏感信息或引入错误的上下文
        self.last_durable_superseded = []       # 上一次模型输出中被识别为具有持久记忆意图但被新信息覆盖或替代的信息列表，这些信息是从模型输出的文本中提取出来的，并且被认为是用户希望保存到持久记忆中的重要信息，但由于后续的模型输出中出现了新的信息，导致这些信息被覆盖或替代，Lincoder 会记录这些信息，并在后续的工具调用和模型交互中避免使用这些信息，以防止引入过时的上下文或错误的信息
        self._last_tool_result_metadata = {}    # 上一次工具执行结果的元数据信息，包含工具执行的输入参数、输出结果、执行日志、可能的异常信息等，这些信息会在评估模型输出时使用，帮助 Lincoder 理解模型的行为和决策过程，并进行相应的工具调用和上下文更新
        self._last_prefix_refresh = {           # 上一次刷新 prompt 前缀时的诊断信息，包含工作区是否发生变化、提示词是否发生变化等信息，这些信息会在 refresh_prefix() 方法中更新，并在后续的工具调用和模型交互中使用，帮助 Lincoder 理解上下文变化对提示词的影响，并进行相应的工具调用和上下文更新
            "workspace_changed": False,
            "prefix_changed": False,
        }

    @classmethod  # 表示这是一个类方法
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        '''
        通过传入外部依赖和会话 ID，自动加载并组装出一个完整的对象实例
        '''

        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs
        )

    def _ensure_session_shape(self):
        """
        确保 session 字典具有正确的结构和必要的字段，如果缺失则进行初始化。

        history：历史对话，确保对话历史是一个列表。这是 Agent 维持上下文记忆的基础。
        memory：长期记忆状态
        checkpoints：检查点/断点续传。包含 current_id 和 items
        runtime_identity & resume_state：运行时身份与恢复状态。用于隔离“业务数据”和“系统元数据”。
        """

        self.session.setdefault("history", [])
        self.session.setdefault("memory", memorylib.default_memory_state())
        checkpoints = self.session.setdefault("checkpoints", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            self.session["checkpoints"] = checkpoints
        checkpoints.setdefault("current_id", "")
        checkpoints.setdefault("items", {})
        runtime_identity = self.session.setdefault("runtime_identity", {})
        if not isinstance(runtime_identity, dict):
            self.session["runtime_identity"] = {}
        resume_state = self.session.setdefault("resume_state", {})
        if not isinstance(resume_state, dict):
            self.session["resume_state"] = {}

    def build_tools(self):
        """
        构建工具列表，返回一个字典，键是工具名称，值是工具对象。
        这里的工具对象需要实现一个 execute(args) 方法，接受参数字典并返回结果字符串。
        """

        return toolkit.build_tool_registry(self.tool_context())

    @staticmethod
    def _normalize_allowed_tools(allowed_tools):
        """
        接收外部传入的工具白名单，进行严格的清洗、校验和类型转换，确保后续执行引擎拿到的工具列表是绝对安全且格式统一的。
        """

        if allowed_tools is None:
            return None

        normalized = tuple(str(name).strip() for name in allowed_tools)     # 将结果转换为元组
        if not normalized or any(not name for name in normalized):
            raise ValueError("allowed_tools must be a non-empty sequence of tool names")
        return normalized

    def _apply_tool_allowlist(self, tools):
        """
        在将工具列表暴露给大语言模型（LLM）之前，根据预设的权限配置，剔除掉 LLM 无权使用的工具
        """

        # 1. 没有配置任何白名单限制，直接返回完整工具列表
        if self.allowed_tools is None:
            return tools

        # 2. 严格的白名单合法性校验
        legal_names = toolkit.legal_tool_names()
        unknow = [name for name in self.allowed_tools if name not in legal_names]
        if unknow:
            raise ValueError(f"unknow allowed tool: {', '.join(unknow)}")

        # 3. 过滤工具列表，只保留在白名单中的工具
        allowd = set(self.allowed_tools)
        return {
            name: tool
            for name, tool in tools.items()
            if name in allowd
        }

    def build_prefix(self):
        return build_prompt_prefix(workspace=self.workspace, tools=self.tools)

    def _apply_prefix_state(self, prefix_state):
        self.prefix_state = prefix_state
        self.prefix = prefix_state.text

    def refresh_prefix(self, force=False):
        """
        系统提示词（System Prefix）增量刷新机制。
        在 AI Agent 运行过程中，智能地判断底层环境（工作区）是否发生了变化，从而决定是否需要重新构建并注入系统提示词。
        """

        # 1. 提取上一次生成的提示词哈希值 和 工作区指纹
        previous_hash = getattr(getattr(self, "prefix_state", None), "hash", None)
        previous_workspace_fingerprint = getattr(getattr(self, "prefix_state", None), "workspace_fingerprint", None)

        # 工作区事实相对稳定，所以这里按整体刷新；
        # 只有这些事实真的变化了，才重建完整 prefix。
        # 2. 重新构建工作区上下文并计算其最新的指纹
        refreshed_workspace = WorkspaceContext.build(self.root)
        refreshed_workspace_fingerprint = refreshed_workspace.fingerprint()
        # 如果当前指纹与历史指纹不同，或者传入了强制刷新参数（force=True），则判定工作区已变更，并更新内存中的工作区对象。
        workspace_changed = force or refreshed_workspace_fingerprint != previous_workspace_fingerprint
        if workspace_changed:
            self.workspace = refreshed_workspace

        # 3. 按需重建提示词：
        #   只有当工作区发生变化、强制刷新、或这是首次初始化时，才真正调用 build_prefix() 构建新的提示词状态；否则，直接复用旧的 self.prefix_state。
        prefix_state = self.build_prefix() if workspace_changed or force or previous_hash is None else self.prefix_state
        prefix_changed = force or previous_hash != prefix_state.hash
        if prefix_changed:
            self._apply_prefix_state(prefix_state)

        # 4. 记录并返回刷新诊断信息
        self._last_prefix_refresh = {
            "workspace_changed": workspace_changed,
            "prefix_changed": prefix_changed,
        }
        return dict(self._last_prefix_refresh)

    def feature_enabled(self, name):
        return bool(self.feature_flags.get(str(name), False))

    def _build_prompt_and_metadata(self, user_message):
        """
        Prompt 组装与全链路可观测性（Observability）机制。
        在每一轮对话开始前，不仅负责将各个上下文模块拼装成最终的 Prompt，还极其详尽地记录了“这轮 Prompt 是如何被构建出来的”以及“当前的系统状态”。
        """

        # 1. 触发状态刷新与恢复评估
        refresh = self.refresh_prefix()
        # 调用 evaluate_resume_state() 评估当前是否处于断点续传（Resume）状态
        self.resume_state = self.evaluate_resume_state()

        # 2. 核心 Prompt 构建
        prompt, metadata = self.context_manager.build(user_message)

        # 3. 注入“组装过程”的量化指标（Token 成本分析）
        # 这里把“这轮 prompt 是怎么拼出来的”连同缓存相关状态一起记下来，
        # 后面 trace/report 才能解释清楚：为什么这一轮 prefix 变了、缓存有没有命中。
        metadata.update(
            {
                "prefix_chars": len(self.prefix),
                "workspace_chars": len(self.workspace.text()),
                "memory_chars": len(self.memory_text()),
                "history_chars": len(self.history_text()),
                "request_chars": len(user_message),
                "tool_count": len(self.tools),
                "workspace_docs": len(self.workspace.project_docs),
                "recent_commits": len(self.workspace.recent_commits),
                "prefix_hash": self.prefix_state.hash,
                "prompt_cache_key": self.prefix_state.hash,
                "workspace_fingerprint": self.prefix_state.workspace_fingerprint,
                "tool_signature": self.prefix_state.tool_signature,
                "workspace_changed": refresh["workspace_changed"],
                "prefix_changed": refresh["prefix_changed"],
                "prompt_cache_supported": bool(getattr(self.model_client, "supports_prompt_cache", False)),
                "resume_status": self.resume_state.get("status", CHECKPOINT_NONE_STATUS),
                "stale_summary_invalidations": int(self.resume_state.get("stale_summary_invalidations", 0)),
                "stale_paths": list(self.resume_state.get("stale_paths", [])),
                "runtime_identity_mismatch_fields": list(self.resume_state.get("runtime_identity_mismatch_fields", [])),
            }
        )
        metadata.update(self.detected_secret_env_summary())
        return prompt, metadata

    def capture_workspace_snapshot(self):
        """
        工作区文件快照（Workspace Snapshot）功能
        在 Agent 执行任务前后，快速扫描并记录当前工作区所有文件的相对路径和内容哈希值（SHA-256）
        """

        snapshot = {}
        for path in self.root.rglob("*"):   # 使用 rglob("*") 递归遍历根目录下的所有文件和文件夹
            try:
                # 在计算相对路径时，如果某些符号链接（Symlink）指向了工作区之外的路径，relative_to 会抛出 ValueError
                relative_parts = path.relative_to(self.root).parts
            except ValueError:
                continue

            # 检查路径的任何一部分（文件夹名或文件名）是否命中了 IGNORED_PATH_NAMES（如 .git, __pycache__ 等）。
            if any(part in IGNORED_PATH_NAMES for part in relative_parts):
                continue
            # 跳过所有非文件条目
            if not path.is_file():
                continue

            # 核心指纹生成（SHA-256）
            try:
                # 读取文件的二进制内容，计算其 SHA-256 哈希值，并以相对路径（统一使用正斜杠 /）为键存入 snapshot 字典。
                snapshot[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    @staticmethod
    def diff_workspace_snapshots(before, after):
        """
        工作区状态对比器
        接收 Agent 执行任务前后的两份工作区快照（即包含文件路径和 SHA-256 哈希值的字典），通过对比哈希值，精准地计算出哪些文件被创建、修改或删除了。
        """
        changed_paths = []
        summaries = []
        # 提取并排序全量文件路径
        all_paths = sorted(set(before) | set(after))

        for path in all_paths:
            if before.get(path) == after.get(path):
                continue
            changed_paths.append(path)

            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed_paths, summaries

    def update_memory_after_tool(self, name, args, result):
        """把少量高价值工具结果沉淀到 working memory。

        为什么存在：
        并不是每个工具结果都值得长期带进下一轮 prompt。
        完整结果已经进了 `history`，这里只挑少量“下一轮大概率还会用到”的事实做提纯，
        例如最近读写过哪些文件、某个文件读出来的短摘要。

        输入 / 输出：
        - 输入：工具名 `name`、参数 `args`、执行结果 `result`
        - 输出：无显式返回值，副作用是更新 `self.memory`

        在 agent 链路里的位置：
        它发生在 `run_tool()` 真正执行完工具之后、下一轮 prompt 组装之前。
        也就是说：工具结果先进入完整历史，再由这个函数择优沉淀成轻量记忆。
        """

        # 绝大多数工具（如 run_shell, search）的结果不需要进入结构化记忆，只有与文件强相关的操作才值得被记住。
        # 首先检查系统是否开启了“记忆”功能
        if not self.feature_enabled("memory"):
            return
        # 提取工具参数中的 path，如果没有路径则直接返回。
        path = args.get("path")
        if not path:
            return

        # 路径标准化
        canonical_path = self.memory.canonical_path(path)

        # 只要涉及文件的读取、写入或打补丁，就将该文件标记为“活跃/已知文件”。让 Agent 在下一轮对话中，清楚地知道自己最近操作过哪些文件。
        if name in {"read_file", "write_file", "patch_file"}:
            self.memory.remember_file(canonical_path)
        # 当工具是 read_file 时，调用 summarize_read_result 将几百行的文件内容压缩成几句核心摘要。
        # 然后将摘要存入文件专属记忆区，并作为一条带标签的笔记（Note）追加到全局工作记忆中。
        if name == "read_file":
            summary = memorylib.summarize_read_result(result)
            self.memory.set_file_summary(canonical_path, summary)
            self.memory.append_note(summary, tags=(canonical_path,), source=canonical_path)
        # 当文件被修改时，主动将之前为该文件生成的摘要标记为“失效（Invalidated）”。
        # 用于“防止幻觉”。如果 Agent 修改了代码，但工作记忆里还保留着修改前的旧摘要，LLM 在后续推理时就会基于过期的信息做出错误判断。
        elif name in {"write_file", "patch_file"}:
            self.memory.invalidate_file_summary(canonical_path)

    def record_process_note_for_tool(self, name, metadata):
        """
        工具执行异常/状态的过程记录机制（Process Note Recorder）。
        当工具执行出现非正常状态（如部分成功、报错或被拒绝）时，自动提取关键信息，生成一条带有明确指导意义的“过程笔记（Process Note）”，并将其持久化到工作记忆中。
        """

        # 1. 状态过滤（只记录异常）
        status = str(metadata.get("tool_status", "")).strip()
        if status not in {"partial_success", "error", "rejected"}:
            return

        # 2. 从元数据中提取出本次操作涉及的文件路径列表，过滤掉空字符串。然后将它们拼接成逗号分隔的字符串；如果没有任何路径，则默认使用 "workspace"（工作区）作为兜底文本。
        affected_paths = [str(path).strip() for path in metadata.get("affected_paths", []) if str(path).strip()]
        path_text = ", ".join(affected_paths) or "workspace"

        # 3.  生成带“行动指导”的提示文本
        if status == "partial_success":
            # 部分成功：提示 LLM 在重试前先检查差异
            text = f"{name} partial_success on {path_text}; inspect diff before retry"
        elif status == "error":
            # 报错：提示 LLM 先排查失败原因
            text = f"{name} error on {path_text}; check the failure before retry"
        else:
            # 被拒绝：提示 LLM 必须更换其他操作
            text = f"{name} rejected; choose a different action before retry"

        # 标签化与记忆持久化
        tags = ["process", status, *affected_paths]
        self.memory.append_note(text, tags=tuple(tags), source=name, kind="process")
        self.session["memory"] = self.memory.to_dict()

    def ask(self, user_message):
        from lincoder.agent_loop import AgentLoop

        return AgentLoop(self).run(user_message)

    def execute_tool(self, name, args):
        """
        工具执行的核心接口，所有工具调用最终都会走到这里来。
        包括用户直接调用工具（如通过 API）和模型调用工具两种情况。
        """

        result = self.tool_executor.execute(name, args)
        self._last_tool_result_metadata = dict(result.metadata)
        return result

    def run_tool(self, name, args):
        """执行一次工具调用，并在执行前后套上完整护栏。

        为什么存在：
        在 agent 系统里，真正危险的不是“模型会不会想调用工具”，而是
        “平台有没有在执行前把边界守住”。这个函数就是工具层的总闸口：
        所有工具调用都必须先经过它，不能让模型直接碰到底层函数。

        输入 / 输出：
        - 输入：工具名 `name`，参数字典 `args`
        - 输出：字符串结果。无论是成功结果还是错误信息，都会统一返回文本，
          这样模型下一轮都能继续消费这份反馈。

        在 agent 链路里的位置：
        它位于 `ask()` 的“模型决定要调用工具”之后，是控制循环里真正把模型
        意图落到外部世界的一步。因此这里串起了几乎所有安全与可控设计：
        工具是否存在、参数是否合法、是否重复、是否需要审批、执行结果是否裁剪、
        是否需要回写记忆。
        """

        return self.execute_tool(name, args).content

    def repeated_tool_call(self, name, args):
        """
        agent 很常见的一种坏循环，是在没有新信息的情况下反复发起同一调用。
        这里提前挡掉最简单的这种循环。
        """
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        # 死循环通常是连续发生的，因此只需要检查最近的两次调用即可，无需遍历整个历史
        recent = tool_events[-2:]
        # all(...)：只有当最近两次调用的名称和参数都与当前请求完全相同时，才返回 True。
        return all(item["name"] == name and item["args"] == args for item in recent)

    def tool_example(self, name):
        """提供工具调用示例，帮助模型理解工具的正确使用方式。"""
        return toolkit.tool_example(name)

    def validate_tool(self, name, args):
        """把通用工具校验和 runtime 级额外约束串起来。"""
        toolkit.validate_tool(self.tool_context(), name, args)

    def spawn_delegate(self, args):
        """
        子代理生成器（Sub-Agent Spawner）。
        当主 Agent 遇到复杂任务或需要隔离执行时，动态创建一个受限的“子代理（Child Agent）”来专门处理该任务。
        """

        task = str(args.get("task", "")).strip()
        child = Lincoder(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            run_store=self.run_store,
            approval_policy="never",    # 子代理被禁止执行任何需要人工审批的操作，或者在某些框架中意味着禁止执行任何写入操作。
            max_steps=int(args.get("max_steps", 3)),
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=True,             # 强制只读模式。
            secret_env_names=self.secret_env_names,
            shell_env_allowlist=self.shell_env_allowlist,
        )

    def approve(self, name, args):
        """
        在执行具有破坏性或高风险的工具（如 run_shell、write_file）之前，根据系统预设的安全策略，决定是自动放行、直接拦截，还是暂停执行并等待人类的明确授权。
        """

        # 如果当前 Agent 被配置为“只读模式（Read-Only）”，则无条件拒绝任何工具调用。
        # 作用：在进行代码审查、日志分析或排查问题时，开启只读模式可以彻底杜绝 AI 意外修改或删除文件的风险。
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False

        # 当策略既不是 auto 也不是 never 时（例如 manual 模式），程序会暂停执行，通过终端向用户发起交互式询问。
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False

        return answer.strip().lower() in {"y", "yes"}

    def path(self, raw_path):
        """
        路径沙箱（Path Sandbox）机制
        作为所有文件操作工具的“安检门”，确保 AI Agent 只能访问和修改工作区（Workspace Root）内的文件，彻底杜绝路径遍历攻击和符号链接逃逸。
        """

        path = Path(raw_path)
        # 相对路径先挂到仓库根目录下
        path = path if path.is_absolute() else self.root / path

        # 真实物理路径解析：调用 resolve() 方法，解析路径中所有的符号链接（Symlinks）、相对路径符号（. 和 ..），并返回一个标准化的绝对物理路径。
        resolved = path.resolve()

        # 计算 self.root（工作区根目录）和 resolved（解析后的真实路径）的最长公共前缀（commonpath）。
        # 如果这个公共前缀不等于 self.root，说明真实路径已经跑到了工作区之外。
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise ValueError(f"path escapes workspace: {raw_path}")

        return resolved