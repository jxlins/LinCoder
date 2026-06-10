import time
from .task_state import TaskState


class AgentLoop:
    def __init__(self, agent):
        self.agent = agent

    def run(self, user_message):

        agent = self.agent
        run_started_at = time.monotonic()   # 获取单调时钟（Monotonic Clock）值的函数。返回值是一个以小数秒为单位的浮点数

        # 1. 把用户请求写入 memory 的 task_summary，用于后续 prompt 构造，让模型始终知道当前任务目标。
        agent.memory.set_task_summary(user_message)
        # 2. 把用户消息写入历史记录 session history
        agent.record({
            "role": "user", "content": user_message, "created_at": now()
        })

        # 3. 创建任务状态 TaskState
        task_state = TaskState.create(
            run_id=agent.new_run_id(),
            task_id=agent.new_task_id(),
            user_request=user_message
        )
        # 支持 任务恢复机制 可以根据 resume_status 判断是否继续执行，是否存在状态过期，是否存在工作区不一致。
        task_state.resume_status = agent.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        agent.current_task_status = task_state
        # 4. 在 .lincoder/runs/<run_id>/ 下创建本轮 run 目录
        # 5. 立刻写出第一版 task_state.json
        agent.current_run_dir = agent.run_store.start_run(task_state)
        # 6. 追加一条 run_started trace
        agent.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300)
            },
        )

        tool_steps = 0
        attempts = 0
        max_attempts = max(agent.max_steps * 3, agent.max_steps + 4)

        """
        agent 主循环，感知 -> 决策 -> 行动 -> 记录
            1. 感知：重新组 prompt，把当前状态整理给模型看
            2. 决策：让模型返回一个工具调用，或一个最终答案
            3. 行动：如果是工具调用，就执行工具
            4. 记录：把结果写回 history / task_state / trace / memory
            然后进入下一轮，直到停机条件满足
        """

        while tool_steps < agent.max_steps and attempts < max_attempts:

            # 记录一次模型尝试，把当前状态写入磁盘或运行存储
            attempts += 1
            task_state.record_attempt()
            agent.run_store.write_task_state(task_state)

            prompt_started_at = time.monotonic()
            # 构造 prompt
            prompt, prompt_metadata = agent._build_prompt_and_metadata(user_message)
            agent.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000)
                }
            )

            """
            checkpoint 检查点机制：
              1. freshness_mismatch：表示当前恢复状态存在部分过期。比如上一次运行保存的工作区信息和现在不完全一致，或者上下文摘要已经不能完全代表当前状态。
              2. workspace_mismatch：表示工作区身份不一致
              3. context_reduction：表示 prompt 构造时发生了上下文压缩或裁剪。
            """
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="freshness_mismatch")
                agent.run_store.wrote_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "freshness_mismatch"
                    }
                )
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                agent.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", [])),
                    },
                )
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="workspace_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "workspace_mismatch",
                    },
                )
            if prompt_metadata.get("budget_reductions"):
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="context_reduction")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "context_reduction",
                    },
                )
            agent.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )

            prompt_cache_key = None
            prompt_cache_retention = None
            # 判断模型后端是否支持 prompt cache。调模型之前，会先看当前 client 支不支持 prompt cache。
            if getattr(agent.model_client, "supports_prompt_cache", False):
                # 只有后端明确支持时，才把稳定前缀的 hash 作为 cache key 发出去。
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            model_started_at = time.monotonic()

            # 调用大模型
            raw = agent.model_client.complete(
                prompt,
                agent.max_new_tokens,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
            completion_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
                # 把后端返回的 usage/cache 统计并回 prompt_metadata，
                # 方便统一写入 report 和 trace。
                prompt_metadata.update(completion_metadata)
            agent.last_completion_metadata = completion_metadata
            agent.last_prompt_metadata = prompt_metadata

            # 解析模型输出
            kind, payload = agent.parse(raw)
            agent.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )

            # 工具调用分支
            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                task_state.record_tool(name)
                tool_started_at = time.monotonic()
                tool_result = agent.execute_tool(name, args)
                result = tool_result.content

                # 执行完成后，把工具结果写回历史
                agent.record(
                    {
                        "role": "tool",
                        "name": name,
                        "args": args,
                        "content": result,
                        "created_at": now(),
                    }
                )
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "tool_executed",
                    {
                        "name": name,
                        "args": args,
                        "result": clip(result, 500),
                        "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                        **dict(tool_result.metadata or {}),
                    },
                )

                # 创建 checkpoint
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="tool_executed")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_executed",
                    },
                )
                continue

            # retry 分支，表示模型输出不符合预期，但是系统还没放弃
            if kind == "retry":
                agent.record({"role": "assistant", "content": payload, "created_at": now()})
                agent.run_store.write_task_state(task_state)
                continue

            # 最终答案分支
            final = (payload or raw).strip()
            agent.record({"role": "assistant", "content": final, "created_at": now()})
            # 标记任务成功完成
            task_state.finish_success(final)
            # 把本次任务中值得长期保存的信息提升到 durable memory
            agent.promote_durable_memory(user_message, final)
            # 创建最终 checkpoint
            checkpoint = agent.create_checkpoint(task_state, user_message, trigger="run_finished")
            agent.run_store.write_task_state(task_state)
            agent.emit_trace(
                task_state,
                "checkpoint_created",
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "trigger": "run_finished",
                },
            )
            agent.emit_trace(
                task_state,
                "run_finished",
                {
                    "status": task_state.status,
                    "stop_reason": task_state.stop_reason,
                    "final_answer": final,
                    "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                },
            )
            # 写入 trace 和 report
            agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
            return final

        # 停止条件
        if attempts >= max_attempts and tool_steps < agent.max_steps:
            # 模型多次输出无效内容，没有产生有效工具调用或最终答案。
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            task_state.stop_retry_limit(final)

        else:
            # 工具调用次数达到了上限，但仍然没有最终答案。
            final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)

        agent.record({"role": "assistant", "content": final, "created_at": now()})
        agent.promote_durable_memory(user_message, final)
        agent.run_store.write_task_state(task_state)
        checkpoint = agent.create_checkpoint(task_state, user_message, trigger=task_state.stop_reason or "run_stopped")
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_stopped",
            },
        )
        agent.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
        return final


