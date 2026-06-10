"""
一次 run() 运行过程中的状态机快照。

它回答的是：这次用户请求当前进行到哪了、调了多少次工具、最后为什么停下。
这个对象会被不断写入 task_state.json，供运行中观察和运行后复盘。
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

STATUS_RUNNING = "running"  # 正在运行中，尚未完成
STATUS_COMPLETED = "completed"  # 已经完成，正常结束
STATUS_STOPPED = "stopped"  # 已经停止，非正常结束，可能是因为达到工具调用限制、时间限制、用户中断等原因
STATUS_FAILED = "failed"  # 已经失败，非正常结束，可能是因为工具调用过程中发生了未处理的异常、模型生成过程中发生了未处理的异常等原因

STOP_REASON_FINAL_ANSWER_RETURNED = "final_answer_returned"
STOP_REASON_STEP_LIMIT_REACHED = "step_limit_reached"
STOP_REASON_RETRY_LIMIT_REACHED = "retry_limit_reached"
STOP_REASON_MODEL_ERROR = "model_error"
STOP_REASON_TOOL_TIMEOUT = "tool_timeout"
STOP_REASON_APPROVAL_DENIED = "approval_denied"
STOP_REASON_DELEGATE_FAILED = "delegate_failed"
STOP_REASON_PERSISTENCE_ERROR = "persistence_error"
STOP_REASON_RESUME_LOAD_ERROR = "resume_load_error"

@dataclass  # 数据类装饰器，
class TaskState:
    """
    追踪与溯源：run_id（单次运行标识）和 task_id（任务标识）分离。一个 Task 可能因为失败被重试多次，每次重试都是一个独立的 Run。
    执行度量：
        attempts：记录模型被调用的轮数（用于防止无限循环）。
        tool_steps：记录真正执行的工具调用次数（用于计费和限制工具滥用）。
    断点续传：checkpoint_id 和 resume_status 配合 session["checkpoints"]，使得长任务在中断后可以从特定节点恢复。
    """

    run_id: str         # 单次运行标识，每次 run() 调用都会生成一个新的 run_id，用于区分不同的运行实例
    task_id: str        # 任务标识，表示当前运行的任务类型或名称，通常与用户请求的内容相关，用于帮助理解当前运行的上下文和目的
    user_request: str
    status: str = STATUS_RUNNING
    tool_steps: int = 0
    attempts: int = 0
    last_tool: str = ""
    stop_reason: str = ""
    final_answer: str = ""
    checkpoint_id: str = ""
    resume_status: str = ""

    @classmethod
    def create(cls, task_id, user_request, run_id=""):
        """
        自动生成带有时间戳和 UUID 前缀的 run_id
        """

        if not run_id:
            run_id = "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]

        # cls(...) 实际上等价于调用 TaskState(...)
        return cls(run_id=run_id, task_id=task_id, user_request=user_request)

    @classmethod
    def from_dict(cls, data):
        """
        提供极其健壮的反序列化能力
        """

        return cls(
            run_id=str(data.get("run_id", "")),
            task_id=str(data.get("task_id", "")),
            user_request=str(data.get("user_request", "")),
            status=str(data.get("status", STATUS_RUNNING)),
            tool_steps=int(data.get("tool_steps", 0)),
            attempts=int(data.get("attempts", 0)),
            last_tool=str(data.get("last_tool", "")),
            stop_reason=str(data.get("stop_reason", "")),
            final_answer=str(data.get("final_answer", "")),
            checkpoint_id=str(data.get("checkpoint_id", "")),
            resume_status=str(data.get("resume_status", "")),
        )

    def record_attempt(self):
        """
        attempt 统计的是“模型被调用了几轮”，不等于 tool_steps。
        """

        self.attempts += 1
        return self

    def record_tool(self, name):
        """
        tool_steps 只统计真正进入执行阶段的工具调用次数。
        """

        self.tool_steps += 1
        self.last_tool = str(name or "")
        return self

    def stop(self, stop_reason, status=STATUS_STOPPED, final_answer=""):
        """
        stop_reason 和 status 分开存，是为了区分“怎么停的”和“停下时是什么状态”。
        """

        self.status = status
        self.stop_reason = stop_reason
        if final_answer != "":
            self.final_answer = final_answer
        return self

    def stop_step_limit(self, final_answer=""):
        return self.stop(STOP_REASON_STEP_LIMIT_REACHED, final_answer=final_answer)

    def stop_retry_limit(self, final_answer=""):
        return self.stop(STOP_REASON_RETRY_LIMIT_REACHED, final_answer=final_answer)

    def stop_model_error(self, final_answer=""):
        return self.stop(STOP_REASON_MODEL_ERROR, status=STATUS_FAILED, final_answer=final_answer)

    def finish_success(self, final_answer):
        self.status = STATUS_COMPLETED
        self.stop_reason = STOP_REASON_FINAL_ANSWER_RETURNED
        self.final_answer = str(final_answer)
        return self

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "user_request": self.user_request,
            "status": self.status,
            "tool_steps": self.tool_steps,
            "attempts": self.attempts,
            "last_tool": self.last_tool,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "checkpoint_id": self.checkpoint_id,
            "resume_status": self.resume_status,
        }