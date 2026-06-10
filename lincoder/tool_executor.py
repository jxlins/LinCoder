"""Structured tool execution for the agent runtime."""

from dataclasses import dataclass
import re

from .workspace import clip

@dataclass(frozen=True)   # frozen=True 表示这个对象创建后不可修改，有利于保证工具执行结果的稳定性，避免后续流程意外改写。
class ToolExecutionResult:
    """
    一次工具调用的结果
        content：给模型看的文本结果
        metadata：给 runtime、trace、report 使用的结构化信息
    """
    content: str
    metadata: dict

def _metadata(
        tool_status,
        tool_error_code="",
        security_event_type="",
        risk_level="low",
        read_only=True,
        affected_paths=None,
        workspace_changed=False,
        workspace_fingerprint="",
        diff_summary=None,
):
    """
    生成标准化工具元数据
    把每次工具调用的状态统一成一个字典

    :param tool_status: 工具状态，例如 ok、error、rejected、partial_success
    :param tool_error_code: 错误类型，例如 unknown_tool、invalid_arguments
    :param security_event_type: 安全事件类型，例如 path_escape
    :param risk_level: 风险等级
    :param read_only: 本次工具是否只读
    :param affected_paths: 受影响的文件路径
    :param workspace_changed: 工作区是否发生变化
    :param workspace_fingerprint: 文件变更摘要
    :param diff_summary: 工作区指纹，只有传入时才写入
    :return:
    """
    result = {
        "tool_status": tool_status,
        "tool_error_code": tool_error_code,
        "security_event_type": security_event_type,
        "risk_level": risk_level,
        "read_only": read_only,
        "affected_paths": list(affected_paths or []),
        "workspace_changed": bool(workspace_changed),
        "diff_summary": list(diff_summary or []),
    }
    if workspace_fingerprint:
        result["workspace_fingerprint"] = workspace_fingerprint
    return result

class ToolExecutor:
    """
    ToolExecutor 持有一个 agent 对象。
    它不自己保存工具列表、审批规则、记忆系统、工作区快照，而是通过 agent 访问这些能力。

    即 ToolExecutor 是执行层，agent 是运行时上下文。
    """
    def __init__(self, agent):
        self.agent=agent

    def execute(self, name, args):
        """
        经过一系列检查：
            检查工具是否在白名单内
            检查工具是否存在
            校验工具参数
            检查是否重复调用
            高风险工具请求审批
            执行前记录工作区快照
            真正运行工具
            执行后记录工作区快照
            比较文件变化
            更新记忆
            记录过程性提示
            返回统一结果
        """

        agent = self.agent

        # 1. 检查工具是否在白名单内
        if agent.allowed_tools is not None and name not in agent.allowed_tools:
            return ToolExecutionResult(
                content=f"error: tool '{name}' is not allowed in this run",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="tool_not_allowed",
                    risk_level="high",
                    read_only=False,
                ),
            )

        # 2. 检查工具是否存在
        tool = agent.tools.get(name)
        if tool is None:
            return ToolExecutionResult(
                content=f"error: unknown tool '{name}'",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="unknown_tool",
                    risk_level="high",
                    read_only=False,
                ),
            )

        # 3. 参数校验
        try:
            agent.validate_tool(name, args)
        except Exception as exc:
            example = agent.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            # 如果该工具有示例，还会把示例拼进去，帮助模型理解正确的参数格式和语义。这是一个重要的提示优化，能显著降低模型调用工具时的格式错误率。
            if example:
                message += f"\nexample: {example}"
            # 识别路径逃逸：系统把路径越界访问视为安全事件。例如模型试图访问工作区外部文件，就会被识别为 path_escape。
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            return ToolExecutionResult(
                content=message,
                metadata=_metadata(
                    "rejected",
                    tool_error_code="invalid_arguments",
                    security_event_type=security_event_type,
                    risk_level="high" if tool["risky"] else "low",
                    read_only=not tool["risky"],
                ),
            )

        # 4. 重复工具调用检测
        if agent.repeated_tool_call(name, args):
            return ToolExecutionResult(
                content=f"error: repeated identical tool call for {name}; choose a different tool or return a final answer",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="repeated_identical_call",
                    risk_level="high" if tool["risky"] else "low",
                    read_only=not tool["risky"],
                ),
            )

        # 5. 高风险工具审批
        if tool["risky"] and not agent.approve(name, args):
            return ToolExecutionResult(
                content=f"error: approval denied for {name}",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="approval_denied",
                    security_event_type="read_only_block" if agent.read_only else "approval_denied",
                    risk_level="high",
                    read_only=False,
                ),
            )

        # 6. 执行前记录工作区快照（如果工具被标记为“高风险”）
        before_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else {}
        after_snapshot = before_snapshot
        try:

            # 7. 真正运行工具
            content = clip(tool["run"](args))

            after_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            # affected_paths：哪些文件变了、diff_summary：created、deleted、modified 摘要、workspace_changed：是否有变化
            affected_paths, diff_summary = agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)

            tool_status = "ok"
            tool_error_code = ""

            # 对于 shell 命令，代码会从输出中解析 exit_code
            if name == "run_shell":
                match = re.search(r"exit_code:\s*(-?\d+)", content)
                exit_code = int(match.group(1)) if match else 0
                # 命令失败，但工作区改变了：partial_success
                if exit_code != 0 and workspace_changed:
                    # partial_success 是一个很重要的状态。
                    # 例如执行脚本时报错，但脚本已经生成了一部分文件，或者 patch 执行到一半。
                    # 这时不能简单当成完全失败，因为工作区已经发生副作用。后续模型需要先检查 diff，再决定如何处理。
                    tool_status = "partial_success"
                    tool_error_code = "tool_partial_success"
                # 命令失败，工作区没变：error
                elif exit_code != 0:
                    tool_status = "error"
                    tool_error_code = "tool_failed"

            # 8. 工具执行成功后把高价值信息更新到 agent 记忆中
            # 解决的问题：完整工具结果进入 history，但 history 可能很长；memory 只保存提炼后的高价值信息，方便后续 prompt 使用。
            agent.update_memory_after_tool(name, args, content)

            # 9. 生成最终 metadata
            metadata = _metadata(
                tool_status,
                tool_error_code=tool_error_code,
                risk_level="high" if tool["risky"] else "low",
                read_only=not tool["risky"],
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary,
            )

            # 10. 记录过程性记忆
            agent.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=content, metadata=metadata)

        except Exception as exc:
            # 异常时仍然会重新捕获工作区快照
            after_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)

            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            metadata = _metadata(
                "partial_success" if workspace_changed else "error",
                tool_error_code="tool_partial_success" if workspace_changed else "tool_failed",
                security_event_type=security_event_type,
                risk_level="high" if tool["risky"] else "low",
                read_only=not tool["risky"],
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary,
            )
            agent.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=f"error: tool {name} failed: {exc}", metadata=metadata)