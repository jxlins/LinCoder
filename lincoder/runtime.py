"""Agent 运行时核心逻辑。

Lincoder 就是包在模型外面的控制循环：负责组 prompt、解析模型输出、
校验并执行工具、写 trace、更新工作记忆，以及在合适的时候停下来。
"""

import re
import uuid
from datetime import datetime
from pathlib import Path

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
        self._last_prefix_refresh = {           # 上一次前缀状态刷新的相关信息，包含刷新时的上下文状态、记忆状态、工具调用信息等，这些信息会在生成 prompt 和评估模型输出时使用，帮助 Lincoder 理解前缀状态的变化和对模型行为的影响，并进行相应的工具调用和上下文更新
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

    def ask(self, user_message):
        from lincoder.agent_loop import AgentLoop

        return AgentLoop(self).run(user_message)