"""
Checkpoint and resume-state helpers.

解决的问题是：
    Agent 中途停止后怎么恢复
    恢复时如何知道文件有没有变
    恢复时如何知道工具、模型、参数有没有变
    如何把上次任务进度告诉模型
    如何形成 checkpoint 链
"""

import uuid

from . import memory as memorylib
from .workspace import clip, now

CHECKPOINT_SCHEMA_VERSION = "phase1-v1"                         # checkpoint 数据结构版本。
CHECKPOINT_NONE_STATUS = "no-checkpoint"                        # 当前 session 中没有可用 checkpoint。
CHECKPOINT_FULL_VALID_STATUS = "full-valid"                     # 表示 checkpoint 完全有效，可以安全参考
CHECKPOINT_PARTIAL_STALE_STATUS = "partial-stale"               # 表示 checkpoint 部分过期。
CHECKPOINT_WORKSPACE_MISMATCH_STATUS = "workspace-mismatch"     # 表示运行环境发生不一致。
CHECKPOINT_SCHEMA_MISMATCH_STATUS = "schema-mismatch"           # 表示 checkpoint 的数据结构版本和当前代码不一致。

# 运行时身份字段：用来判断当前运行环境和 checkpoint 保存时的运行环境是否一致。
RUNTIME_IDENTITY_KEYS = (
    "cwd",                      # 工作目录
    "model",                    # 模型相关信息
    "model_client",
    "approval_policy",          # 工具执行策略
    "read_only",
    "max_steps",
    "max_new_tokens",
    "feature_flags",
    "shell_env_allowlist",
    "workspace_fingerprint",    # 工作区与工具定义
    "tool_signature",
)

def current_runtime_identity(agent):
    """
    生成当前运行时身份
    当前 Agent 的运行环境整理成一个字典
    """
    return {
        "session_id": agent.session.get("id", ""),
        "cwd": str(agent.root),
        "model": str(getattr(agent.model_client, "model", "")),
        "model_client": agent.model_client.__class__.__name__,
        "approval_policy": agent.approval_policy,                   # 审批策略
        "read_only": bool(agent.read_only),
        "max_steps": int(agent.max_steps),
        "max_new_tokens": int(agent.max_new_tokens),
        "feature_flags": dict(agent.feature_flags),
        "shell_env_allowlist": list(agent.shell_env_allowlist),     # shell 环境白名单
        "workspace_fingerprint": getattr(getattr(agent, "prefix_state", None), "workspace_fingerprint",
                                         agent.workspace.fingerprint()),
        "tool_signature": agent.tool_signature(),
    }

def checkpoint_state(agent):
    """获取 session 中的 checkpoint 状态"""
    agent._ensure_session_shape()
    return agent.session["checkpoints"]


def current_checkpoint(agent):
    """获取当前 checkpoint"""
    state = checkpoint_state(agent)
    checkpoint_id = str(state.get("current_id", "")).strip()
    if not checkpoint_id:
        return None
    return state.get("items", {}).get(checkpoint_id)

def evaluate_resume_state(agent):
    """
    判断当前 checkpoint 的恢复状态

    :param agent:
    :return:
        {
            "status": "partial-stale",
            "stale_paths": ["main.py"],
            "runtime_identity_mismatch_fields": [],
            "stale_summary_invalidations": 1,
        }
    """

    # 1. 先读取旧的恢复状态。
    previous_resume_state = dict(agent.session.get("resume_state", {}) or {})
    # 2. 检查 memory 中的 file summaries 是否过期，收集过期的文件
    invalidated = agent.invalidate_stale_memory()

    # 3. 初始化恢复状态
    checkpoint = current_checkpoint(agent)
    status = CHECKPOINT_NONE_STATUS
    stale_paths = list(invalidated)
    mismatch_fields = []

    # 4.1 如果 checkpoint 存在
    if checkpoint:
        # 4.2 先检查 schema version。
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            status = CHECKPOINT_SCHEMA_MISMATCH_STATUS
        else:
            # 4.3 如果 schema 版本一致，就检查 key files。
            for item in checkpoint.get("key_files", []):
                path = str(item.get("path", "")).strip()
                if not path:
                    continue
                expected = item.get("freshness")
                current = memorylib.file_freshness(path, agent.root)
                # 如果当前哈希和 checkpoint 中保存的哈希不一致，就说明文件变了，于是加入 stale_paths。
                if expected != current and path not in stale_paths:
                    stale_paths.append(path)

            # 4.4 运行时身份比较
            # 取 checkpoint 中保存的运行时身份。如果 checkpoint 中没有，就退回使用 session 中的 runtime_identity。
            saved_identity = dict(checkpoint.get("runtime_identity", {}) or agent.session.get("runtime_identity", {}) or {})
            # 生成当前运行时身份。
            current_identity = current_runtime_identity(agent)
            # 只比较 RUNTIME_IDENTITY_KEYS 中列出的字段。如果保存值和当前值不一致，就加入 mismatch_fields。
            for key in RUNTIME_IDENTITY_KEYS:
                if key not in saved_identity:
                    continue
                if saved_identity.get(key) != current_identity.get(key):
                    mismatch_fields.append(key)
            mismatch_fields.sort()

            # 4.5 确定最终 status
            if stale_paths:
                status = CHECKPOINT_PARTIAL_STALE_STATUS        # 文件过期优先
            elif mismatch_fields:
                status = CHECKPOINT_WORKSPACE_MISMATCH_STATUS   # 运行环境不一致其次
            else:
                status = CHECKPOINT_FULL_VALID_STATUS           # 全部一致则 full-valid

    # 5. 生成 resume_state
    resume_state = {
        "status": status,                                           # 当前 checkpoint 状态。
        "stale_paths": stale_paths,                                 # 过期的文件路径列表。
        "runtime_identity_mismatch_fields": mismatch_fields,        # 运行时不一致字段。
        "stale_summary_invalidations": max(                         # 过期摘要数量。
            len(invalidated),
            int(previous_resume_state.get("stale_summary_invalidations", 0))
            if status == CHECKPOINT_PARTIAL_STALE_STATUS
            else 0,
        ),
    }

    # 6. 把 resume_state 保存到 session 中，供后续使用。
    agent.session["resume_state"] = resume_state
    agent.session["runtime_identity"] = current_runtime_identity(agent)
    return resume_state


def render_checkpoint_text(agent):
    """
    把当前 checkpoint 转成一段文本，供 prompt 使用。
    """

    checkpoint = current_checkpoint(agent)
    if not checkpoint:
        return ""
    # 基础 checkpoint 文本
    lines = [
        "Task checkpoint:",
        f"- Resume status: {agent.resume_state.get('status', CHECKPOINT_NONE_STATUS)}",
        f"- Current goal: {checkpoint.get('current_goal', '-') or '-'}",
        f"- Current blocker: {checkpoint.get('current_blocker', '-') or '-'}",
        f"- Next step: {checkpoint.get('next_step', '-') or '-'}",
    ]

    # 把 checkpoint 记录的关键文件列出来。
    key_files = [str(item.get("path", "")).strip() for item in checkpoint.get("key_files", []) if str(item.get("path", "")).strip()]
    lines.append(f"- Key files: {', '.join(key_files) or '-'}")

    # 渲染已完成项和排除项
    if checkpoint.get("completed"):
        lines.append("- Completed: " + " | ".join(str(item) for item in checkpoint.get("completed", [])))
    if checkpoint.get("excluded"):
        lines.append("- Excluded: " + " | ".join(str(item) for item in checkpoint.get("excluded", [])))

    # 如果有过期文件，就告诉模型。
    if agent.resume_state.get("stale_paths"):
        lines.append("- Stale paths: " + ", ".join(agent.resume_state["stale_paths"]))

    # 最后附加 checkpoint 摘要。最后附加 checkpoint 摘要。
    summary = str(checkpoint.get("summary", "")).strip()
    if summary:
        lines.append(f"- Summary: {summary}")

    return "\n".join(lines)


def infer_next_step(task_state):
    """推断恢复后的下一步"""
    if task_state.status == "completed":
        return "No next step recorded."
    if task_state.stop_reason == "step_limit_reached":
        return "Resume from the latest checkpoint and continue the task."
    if task_state.last_tool:
        return f"Decide the next action after {task_state.last_tool}."
    return "Continue the task from the latest checkpoint."


def create_checkpoint(agent, task_state, user_message, trigger):
    """
    创建新 checkpoint
    """

    # 1. 读取 checkpoint 容器和当前 checkpoint
    # 1.1 拿到 session 中的 checkpoints 容器。
    state = checkpoint_state(agent)
    # 1.2 读取当前 checkpoint，作为新 checkpoint 的 parent。
    current = current_checkpoint(agent)
    # 1.3 生成新的 checkpoint id。
    checkpoint_id = "ckpt_" + uuid.uuid4().hex[:8]

    # 2. 收集 key files 和 freshness
    key_files = []
    freshness = {}
    # 2.1 把 memory 中的最近文件当作关键文件。
    for path in agent.memory.to_dict()["working"]["recent_files"]:
        file_freshness = memorylib.file_freshness(path, agent.root)
        freshness[path] = file_freshness
        key_files.append({"path": path, "freshness": file_freshness})

    # 3. 构造 checkpoint 对象
    checkpoint = {
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": current.get("checkpoint_id", "") if current else "",    # 记录上一个 checkpoint id。多个 checkpoint 可以形成链：
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at": now(),
        "current_goal": str(user_message),
        "completed": [task_state.final_answer] if task_state.final_answer else [],
        "excluded": [],
        "current_blocker": "" if str(task_state.stop_reason or "") in ("", "final_answer_returned") else str(task_state.stop_reason),
        "next_step": infer_next_step(task_state),
        "key_files": key_files,
        "freshness": freshness,
        "summary": f"{trigger}: {clip(str(user_message), 120)}",    # 记录 checkpoint 触发原因和用户请求摘要。
        "runtime_identity": current_runtime_identity(agent),
    }

    # 4. 写入 session
    #    把 checkpoint 放入 items
    #    把 current_id 指向新 checkpoint
    #    把 task_state.checkpoint_id 更新成新 id
    #    把 runtime_identity 写入 session
    #    保存 session 到磁盘
    #    返回 checkpoint
    state["items"][checkpoint_id] = checkpoint
    state["current_id"] = checkpoint_id
    task_state.checkpoint_id = checkpoint_id
    agent.session["runtime_identity"] = checkpoint["runtime_identity"]
    agent.session_path = agent.session_store.save(agent.session)
    return checkpoint