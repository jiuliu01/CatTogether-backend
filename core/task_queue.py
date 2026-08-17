"""Background queue for Feishu-triggered coordinator runs."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import uuid

from config import settings
from core import entry
from core.agent_store import agent_store
from core.delegate_registry import delegate_registry
from core.exceptions import RecoverableRunError
from core.orchestrator import orchestrator
from core.output_intent import compact_artifact_final_message, resolve_output_intent
from core.output_safety import sanitize_agent_text
from core.project_store import project_store
from core.registry import materialize, registry
from core.run_store import run_store
from core.run_supervisor import run_supervisor
from core.run_event_store import run_event_store
from core.session_manager import session_manager
from core.workspace import workspace_manager
from integrations.feishu.mapping_store import feishu_mapping_store
from integrations.feishu.progress import FeishuProgressReporter
from integrations.feishu.sender import FeishuSender, feishu_sender
from models.schemas import AgentRun, FeishuBinding, FeishuInboundMessage, RunResult, Thread

@dataclass
class FeishuTask:
    inbound: FeishuInboundMessage
    binding: FeishuBinding
    thread: Thread
    run: AgentRun
    entry_id_override: str | None = None
    invocation_id: str | None = None
    is_resume: bool = False


@dataclass
class DelegationTask:
    run_id: str
    child_id: str
    resume_prompt: str = ""


@dataclass
class ResumeTask:
    run_id: str
    invocation_id: str


@dataclass
class PendingBinding:
    """A group whose first message arrived before workspace binding.

    The original task is held until the user replies with a project directory
    path; ``_handle_pending_path`` then binds the group and queues the task.
    """

    original_inbound: FeishuInboundMessage
    asked_message_id: str
    created_at: datetime


class FeishuTaskQueue:
    def __init__(self, sender: FeishuSender | None = None) -> None:
        self.sender = sender or feishu_sender
        self._queue: asyncio.Queue[FeishuTask | DelegationTask | ResumeTask] = asyncio.Queue(
            maxsize=max(settings.task_queue_size, 1)
        )
        self._workers: list[asyncio.Task] = []
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._queued_keys: set[str] = set()
        self._reporters: dict[str, FeishuProgressReporter] = {}
        # Groups waiting for the user to reply with a workspace path.
        self._pending_bindings: dict[tuple[str, str], PendingBinding] = {}

    @property
    def running(self) -> bool:
        return bool(self._workers)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if self._workers:
            return
        for index in range(max(settings.task_worker_count, 1)):
            self._workers.append(
                asyncio.create_task(self._worker(index), name=f"feishu-task-worker-{index}")
            )

    async def _reporter_for_run(
        self,
        run: AgentRun,
        *,
        entry_agent_name: str = "",
    ) -> FeishuProgressReporter:
        existing = self._reporters.get(run.id)
        if existing is not None:
            return existing

        if not entry_agent_name and run.entry_agent_id:
            agent = await self._resolve_entry_agent(run.project_id, run.entry_agent_id)
            entry_agent_name = getattr(agent, "name", run.entry_agent_id) if agent else run.entry_agent_id

        delivery = run.outputs.get("delivery") if isinstance(run.outputs, dict) else None
        card_message_id = (
            str(delivery.get("card_message_id") or "")
            if isinstance(delivery, dict)
            else ""
        )
        elapsed = max(
            int((datetime.now(timezone.utc) - run.created_at).total_seconds()), 0
        )

        async def on_delivery(snapshot: dict) -> None:
            await run_store.patch_outputs(run.id, {"delivery": snapshot})

        reporter = FeishuProgressReporter(
            self.sender,
            run.trigger_message_id,
            run.id,
            entry_agent_name=entry_agent_name,
            on_delivery=on_delivery,
            existing_card_message_id=card_message_id or None,
            elapsed_seconds=elapsed,
        )
        if card_message_id:
            reporter.restore_events(await run_event_store.list(run.id))
        self._reporters[run.id] = reporter
        return reporter

    async def _pin_run_workspace(self, run: AgentRun) -> str:
        """Make the durable group binding authoritative for every invocation."""
        bindings = await feishu_mapping_store.list_bindings()
        binding = next(
            (item for item in bindings if item.channel_id == run.channel_id and item.enabled),
            None,
        )
        workspace = binding.workspace_dir if binding else run.workspace_dir
        if not workspace and run.project_id:
            project = await project_store.get(run.project_id)
            workspace = project.workspace_dir if project else ""
        if not workspace:
            workspace = workspace_manager.get_dir(run.channel_id)
        workspace_manager.pin(run.channel_id, workspace)
        if run.workspace_dir != workspace:
            await run_store.update(run.id, workspace_dir=workspace)
        return workspace

    async def stop(self) -> None:
        workers, self._workers = self._workers, []
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        if self._reporters:
            await asyncio.gather(
                *(reporter.pause() for reporter in self._reporters.values()),
                return_exceptions=True,
            )

    def _is_isolated_workspace(self, workspace_dir: str) -> bool:
        """True if ``workspace_dir`` is an auto-created scratch dir under
        ``data/workspaces/`` — i.e. the group was never bound to a real
        project directory."""
        try:
            return Path(workspace_dir).resolve().is_relative_to(
                workspace_manager.base_dir().resolve()
            )
        except (OSError, ValueError):
            return False

    async def _create_and_queue(
        self, inbound: FeishuInboundMessage, binding: FeishuBinding
    ) -> AgentRun:
        """Bind thread → run → queue → reply '已收到'. Shared by the normal
        path and the post-binding path so they can't drift apart."""
        thread = await session_manager.get_or_create_thread(
            binding.channel_id,
            inbound.external_root_id,
            source="feishu",
        )
        run = await run_store.create(
            binding.channel_id,
            thread.id,
            inbound.message_id,
            inbound.external_user_id,
        )
        await run_store.update(
            run.id,
            original_task=inbound.content,
            project_id=binding.project_id,
            workspace_dir=binding.workspace_dir,
            tenant_key=inbound.tenant_key,
            chat_id=inbound.chat_id,
            sender_open_id=inbound.sender_open_id,
        )
        task = FeishuTask(inbound=inbound, binding=binding, thread=thread, run=run)
        try:
            self._queue.put_nowait(task)
        except asyncio.QueueFull:
            await run_store.update(
                run.id,
                status="failed",
                error="任务队列已满，请稍后重试。",
            )
            asyncio.create_task(
                self._safe_reply(
                    inbound.message_id,
                    f"任务未进入队列：当前任务过多，请稍后重试。\nRun ID: {run.id}",
                    f"{run.id}-queue-full",
                )
            )
            return run

        asyncio.create_task(
            self._safe_reply(
                inbound.message_id,
                f"已收到，正在拆解任务。\nRun ID: {run.id}",
                f"{run.id}-accepted",
            )
        )
        return run

    async def _ask_for_workspace(
        self, inbound: FeishuInboundMessage, key: tuple[str, str]
    ) -> None:
        """First message from an unbound group: hold the task and ask the user
        for a project directory path."""
        self._pending_bindings[key] = PendingBinding(
            original_inbound=inbound,
            asked_message_id=inbound.message_id,
            created_at=datetime.now(timezone.utc),
        )
        roots = "\n".join(f"- {r}" for r in workspace_manager.allowed_roots())
        asyncio.create_task(
            self._safe_reply(
                inbound.message_id,
                "当前群未绑定工作空间。请回复要绑定的项目目录绝对路径"
                f"（需在以下根目录下：\n{roots}\n）。"
                "例如 `D:\\Project\\CatTogether`。"
                "绑定后将自动开始处理你刚才的任务。",
                f"pending-binding-{inbound.message_id}",
            )
        )

    async def _handle_pending_path(
        self,
        inbound: FeishuInboundMessage,
        existing: FeishuBinding | None,
        pending: PendingBinding,
        key: tuple[str, str],
    ) -> AgentRun | None:
        """The user replied to a binding prompt; treat this message as a path."""
        raw_path = inbound.content.strip().strip("`\"'").strip()
        try:
            validated = workspace_manager.validate_path(raw_path)
        except ValueError as exc:
            asyncio.create_task(
                self._safe_reply(
                    inbound.message_id,
                    f"路径无效：{exc}\n请重新输入要绑定的项目目录绝对路径。",
                    f"pending-binding-retry-{inbound.message_id}",
                )
            )
            return None

        try:
            binding = await feishu_mapping_store.bind(
                inbound.tenant_key,
                inbound.chat_id,
                str(validated),
                channel_name=None,
            )
        except Exception as exc:
            asyncio.create_task(
                self._safe_reply(
                    inbound.message_id,
                    f"绑定失败：{exc}\n请重新输入要绑定的项目目录绝对路径。",
                    f"pending-binding-fail-{inbound.message_id}",
                )
            )
            return None

        self._pending_bindings.pop(key, None)
        original = pending.original_inbound
        asyncio.create_task(
            self._safe_reply(
                inbound.message_id,
                f"已绑定到 {validated}，开始处理你之前的任务："
                f"{original.content}",
                f"pending-binding-ok-{inbound.message_id}",
            )
        )
        # Queue the original task (the one that triggered the prompt) under
        # the freshly bound workspace. Its inbound carries the original
        # message_id, so the '已收到' reply threads under the first message.
        return await self._create_and_queue(original, binding)

    async def enqueue(self, inbound: FeishuInboundMessage) -> AgentRun | None:
        key = (inbound.tenant_key, inbound.chat_id)
        existing = await feishu_mapping_store.get_binding(
            inbound.tenant_key, inbound.chat_id
        )
        pending = self._pending_bindings.get(key)

        # Branch 1: waiting for a path — this message is the path.
        if pending is not None:
            return await self._handle_pending_path(inbound, existing, pending, key)

        # Branch 2: unbound (no binding, or stuck on an isolated scratch dir).
        if existing is None or self._is_isolated_workspace(existing.workspace_dir):
            await self._ask_for_workspace(inbound, key)
            return None

        # Branch 3: bound — normal flow.
        binding = await feishu_mapping_store.ensure_binding(inbound)
        return await self._create_and_queue(inbound, binding)

    async def enqueue_delegation(self, run_id: str, child_id: str) -> bool:
        key = f"child:{run_id}:{child_id}"
        if key in self._queued_keys:
            return True
        try:
            self._queue.put_nowait(DelegationTask(run_id=run_id, child_id=child_id))
        except asyncio.QueueFull:
            return False
        self._queued_keys.add(key)
        return True

    async def enqueue_resume(self, run_id: str, invocation_id: str) -> bool:
        key = f"resume:{run_id}:{invocation_id}"
        if key in self._queued_keys:
            return True
        try:
            self._queue.put_nowait(
                ResumeTask(run_id=run_id, invocation_id=invocation_id)
            )
        except asyncio.QueueFull:
            return False
        self._queued_keys.add(key)
        return True

    async def _worker(self, worker_index: int) -> None:
        while True:
            task = await self._queue.get()
            if isinstance(task, FeishuTask):
                lock_key = task.thread.id
                queued_key = ""
            elif isinstance(task, DelegationTask):
                lock_key = f"child:{task.run_id}:{task.child_id}"
                queued_key = lock_key
            else:
                lock_key = f"resume:{task.run_id}:{task.invocation_id}"
                queued_key = lock_key
            lock = self._thread_locks.setdefault(lock_key, asyncio.Lock())
            try:
                async with lock:
                    if isinstance(task, FeishuTask):
                        await self._execute(task)
                    elif isinstance(task, DelegationTask):
                        await self._execute_delegation(task)
                    else:
                        await self._execute_resume(task)
            finally:
                if queued_key:
                    self._queued_keys.discard(queued_key)
                self._queue.task_done()

    async def _resolve_targets(self, project_id: str | None, target_ids: list[str]) -> list:
        """Turn routed agent_ids into executable BaseAgent instances.

        Project agent specs are materialized into ClaudeCodeAgent; the enabled
        built-in Claude Code agent comes from the global registry.
        """
        from agents.base import BaseAgent

        resolved: list[BaseAgent] = []
        for aid in target_ids:
            if project_id:
                spec = await agent_store.get(project_id, aid)
                if spec:
                    resolved.append(materialize(spec))
                    continue
            agent = registry.get(aid)
            if agent:
                resolved.append(agent)
        return resolved

    async def _resolve_entry_agent(self, project_id: str | None, entry_id: str) -> object:
        """Materialize the entry agent from its project spec, else registry."""
        from agents.base import BaseAgent

        if project_id:
            spec = await agent_store.get(project_id, entry_id)
            if spec:
                return materialize(spec)
        return registry.get(entry_id)

    async def _enqueue_delegation_later(self, run_id: str, child_id: str) -> None:
        """Requeue after the current worker releases its duplicate-suppression key."""
        await asyncio.sleep(0)
        await self.enqueue_delegation(run_id, child_id)

    async def _maybe_resume_parent(
        self, run_id: str, parent_invocation_id: str
    ) -> None:
        run = await run_store.get(run_id)
        if not run or run.status in ("completed", "failed", "cancelled", "completing"):
            return

        parent = await run_store.get_delegation(run_id, parent_invocation_id)
        if parent is not None:
            if parent.status != "waiting_children":
                return
        elif run.status != "waiting_children":
            return

        if not await run_store.try_mark_resume_queued(run_id, parent_invocation_id):
            return
        if not await self.enqueue_resume(run_id, parent_invocation_id):
            await run_store.clear_resume_queued(run_id, parent_invocation_id)

    @staticmethod
    def _resume_prompt(original_task: str, stage_summary: str, children: list) -> str:
        child_text = "\n\n".join(
            f"子任务 {item.id}（{item.target_agent_id}）\n"
            f"状态：{item.status}\n"
            f"结果：{item.result or item.error or '无返回内容'}"
            for item in children
        )
        return (
            "这是一次任务恢复。请基于已有工作继续，不要重复已经完成的步骤。\n\n"
            f"原始目标：\n{original_task}\n\n"
            f"上一阶段总结：\n{stage_summary or '无'}\n\n"
            f"已完成的子任务：\n{child_text or '无'}\n\n"
            "请验收这些结果，继续剩余工作；若目标已完成，直接给出最终结果。"
        )

    async def _execute_delegation(self, task: DelegationTask) -> None:
        run = await run_store.get(task.run_id)
        child = await run_store.get_delegation(task.run_id, task.child_id)
        if not run or not child:
            return
        if run.status in ("completed", "failed", "cancelled", "completing"):
            await run_store.update_delegation(
                task.run_id, task.child_id, status="cancelled", error="父任务已经结束"
            )
            return

        await self._pin_run_workspace(run)

        agent = await self._resolve_entry_agent(run.project_id, child.target_agent_id)
        if agent is None:
            await run_store.update_delegation(
                task.run_id, task.child_id, status="failed", error="目标角色不可用"
            )
            await self._maybe_resume_parent(task.run_id, child.parent_invocation_id)
            return

        spec = await agent_store.get(run.project_id, child.target_agent_id) if run.project_id else None
        capabilities = list(getattr(spec, "mcp_capabilities", None) or ["delegate", "report_progress", "memory_search"])
        reporter = await self._reporter_for_run(run)
        await reporter.activate()
        child_phase = "coding" if child.access == "write" else "researching"
        await reporter.on_progress(
            "delegate_started",
            {
                "title": f"{getattr(agent, 'name', child.target_agent_id)} 已开始处理子任务",
                "detail": child.task,
                "agent_id": child.target_agent_id,
                "agent_name": getattr(agent, "name", child.target_agent_id),
                "phase": child_phase,
                "child_run_id": child.id,
                "status": "running",
            },
        )
        await run_store.update_delegation(task.run_id, task.child_id, status="running")
        if not await run_supervisor.accepting(task.run_id):
            await run_store.update_delegation(
                task.run_id, task.child_id, status="cancelled", error="父任务已经结束"
            )
            return

        token = await delegate_registry.issue(
            project_id=run.project_id,
            channel_id=run.channel_id,
            thread_id=run.thread_id,
            run_id=run.id,
            depth=child.depth,
            caller_agent_id=child.target_agent_id,
            caller_agent_name=getattr(agent, "name", child.target_agent_id),
            on_progress=reporter.on_progress,
            invocation_id=child.id,
            parent_invocation_id=child.parent_invocation_id,
            mcp_capabilities=capabilities,
            output_intent=run.intent,
            user_id=run.trigger_user_id or "default",
        )
        error: str | None = None
        text = ""
        try:
            async with delegate_registry.acquire():
                text, error = await orchestrator.invoke_for_delegate(
                    agent=agent,
                    content=task.resume_prompt or child.task,
                    channel_id=run.channel_id,
                    thread_id=run.thread_id,
                    project_id=run.project_id,
                    access=child.access,
                    delegate_token=token,
                    mcp_capabilities=capabilities,
                    output_intent=run.intent,
                    on_progress=reporter.on_progress,
                    user_id=run.trigger_user_id or "default",
                    root_run_id=run.id,
                    invocation_id=child.id,
                    parent_invocation_id=child.parent_invocation_id,
                )
        except Exception as exc:
            error = str(exc)
        finally:
            await delegate_registry.revoke(token)

        if error and error.startswith("RECOVERABLE:"):
            attempts = child.recovery_count + 1
            if attempts <= max(settings.recovery_max_attempts, 0):
                await run_store.update_delegation(
                    run.id, child.id, status="recovering",
                    error=error.removeprefix("RECOVERABLE:"), recovery_count=attempts,
                )
                await reporter.on_progress(
                    "stalled",
                    {
                        "title": f"{getattr(agent, 'name', child.target_agent_id)} 暂无进展，正在自动恢复",
                        "detail": error.removeprefix("RECOVERABLE:"),
                        "agent_id": child.target_agent_id,
                        "agent_name": getattr(agent, "name", child.target_agent_id),
                        "phase": child_phase,
                        "child_run_id": child.id,
                        "status": "recovering",
                    },
                )
                asyncio.create_task(self._enqueue_delegation_later(run.id, child.id))
                return
        if error:
            await run_store.update_delegation(
                run.id, child.id, status="failed", error=error
            )
            await reporter.on_progress(
                "delegate_failed",
                {
                    "title": f"{getattr(agent, 'name', child.target_agent_id)} 子任务失败",
                    "detail": error,
                    "agent_id": child.target_agent_id,
                    "agent_name": getattr(agent, "name", child.target_agent_id),
                    "phase": child_phase,
                    "child_run_id": child.id,
                    "status": "failed",
                },
            )
            await self._maybe_resume_parent(run.id, child.parent_invocation_id)
            return

        grandchildren = await run_store.children_for_parent(run.id, child.id)
        if grandchildren:
            await run_store.update_delegation(
                run.id, child.id, status="waiting_children", result=text
            )
            await reporter.on_progress(
                "checkpoint",
                {
                    "title": f"{getattr(agent, 'name', child.target_agent_id)} 正在等待其子任务完成",
                    "detail": text,
                    "agent_id": child.target_agent_id,
                    "agent_name": getattr(agent, "name", child.target_agent_id),
                    "phase": child_phase,
                    "child_run_id": child.id,
                    "status": "waiting_children",
                },
            )
            await self._maybe_resume_parent(run.id, child.id)
            return

        await run_store.update_delegation(
            run.id, child.id, status="completed", result=text or "子任务已完成"
        )
        await reporter.on_progress(
            "delegate_completed",
            {
                "title": f"{getattr(agent, 'name', child.target_agent_id)} 已完成子任务",
                "detail": text,
                "agent_id": child.target_agent_id,
                "agent_name": getattr(agent, "name", child.target_agent_id),
                "phase": child_phase,
                "child_run_id": child.id,
                "status": "completed",
            },
        )
        await self._maybe_resume_parent(run.id, child.parent_invocation_id)

    async def _execute_resume(self, task: ResumeTask) -> None:
        run = await run_store.get(task.run_id)
        if not run or run.status in ("completed", "failed", "cancelled", "completing"):
            return
        children = await run_store.children_for_parent(run.id, task.invocation_id)
        parent = await run_store.get_delegation(run.id, task.invocation_id)
        if (parent is not None and not children) or any(
            child.status not in ("completed", "failed", "cancelled") for child in children
        ):
            await run_store.clear_resume_queued(run.id, task.invocation_id)
            return

        prompt = self._resume_prompt(run.original_task, run.stage_summary, children)
        if children:
            await run_store.mark_children_consumed(run.id, task.invocation_id)
        else:
            await run_store.clear_resume_queued(run.id, task.invocation_id)
        if parent is not None:
            await run_store.update_delegation(run.id, parent.id, status="recovering")
            await self._execute_delegation(
                DelegationTask(run_id=run.id, child_id=parent.id, resume_prompt=prompt)
            )
            return


        bindings = await feishu_mapping_store.list_bindings()
        binding = next((item for item in bindings if item.channel_id == run.channel_id), None)
        thread = await session_manager.get_thread(run.thread_id)
        if binding is None or thread is None:
            await run_store.update(run.id, status="failed", error="恢复任务时找不到原群聊或会话")
            return
        workspace_manager.pin(run.channel_id, run.workspace_dir or binding.workspace_dir)
        inbound = FeishuInboundMessage(
            message_id=run.trigger_message_id,
            tenant_key=run.tenant_key or binding.tenant_key,
            chat_id=run.chat_id or binding.chat_id,
            sender_open_id=run.sender_open_id or "recovery",
            root_id=thread.external_root_id,
            content=prompt,
        )
        await self._execute(
            FeishuTask(
                inbound=inbound,
                binding=binding,
                thread=thread,
                run=run,
                entry_id_override=run.entry_agent_id or "coordinator",
                invocation_id=uuid.uuid4().hex,
                is_resume=True,
            )
        )

    async def restore_pending(self) -> int:
        """Requeue durable unfinished work after a service restart."""
        restored = 0
        for run in await run_store.list(limit=10_000):
            if run.status in ("completed", "failed", "cancelled") or not run.original_task:
                continue
            await run_supervisor.reopen_run(run.id)
            await self._pin_run_workspace(run)
            active_children = [
                child for child in run.delegations
                if not child.consumed
                and child.status in ("queued", "running", "recovering")
            ]
            for child in active_children:
                await run_store.update_delegation(
                    run.id, child.id, status="recovering",
                    recovery_count=child.recovery_count + 1,
                )
                if await self.enqueue_delegation(run.id, child.id):
                    restored += 1
            waiting_parents = {
                child.parent_invocation_id for child in run.delegations if not child.consumed
            }
            for parent_id in waiting_parents:
                await self._maybe_resume_parent(run.id, parent_id)
            if not active_children and not run.pending_child_ids:
                await run_store.update(
                    run.id, status="recovering", recovery_count=run.recovery_count + 1
                )
                if await self.enqueue_resume(run.id, run.root_invocation_id or "root"):
                    restored += 1
        return restored

    async def _execute(self, task: FeishuTask) -> None:
        reporter: FeishuProgressReporter | None = None
        current_invocation_id = task.invocation_id or uuid.uuid4().hex

        async def on_status(status: str, outputs: dict) -> None:
            plan = {"text": outputs["plan"]} if outputs.get("plan") else None
            await run_store.update(
                task.run.id,
                status=status,
                plan=plan,
                outputs=outputs,
            )

        try:
            await run_supervisor.reopen_run(task.run.id)
            workspace_manager.pin(task.binding.channel_id, task.binding.workspace_dir)
            await run_store.update(
                task.run.id,
                workspace_dir=task.binding.workspace_dir,
                project_id=task.binding.project_id,
            )
            # --- Ensure project + default agents exist for this group ---
            project_id = task.binding.project_id
            if project_id:
                project = await project_store.get(project_id)
                if project:
                    # Seed the four cats if the project has no agents yet, OR
                    # if the specific entry agent is missing (covers projects
                    # created before seed_defaults existed).
                    specs = await agent_store.list(project_id)
                    spec_ids = {s.agent_id for s in specs}
                    if not spec_ids:
                        await agent_store.seed_defaults(project_id)
                    elif project_id and "coordinator" not in spec_ids:
                        await agent_store.seed_defaults(project_id)

            # --- Entry resolution: who takes the task? ---
            if task.entry_id_override:
                entry_id, task_text = task.entry_id_override, task.inbound.content
            else:
                entry_id, task_text = await entry.resolve(
                    task.inbound.content, project_id=project_id
                )
            delivery_intent = resolve_output_intent(task_text)
            await run_store.update(
                task.run.id,
                intent=delivery_intent.as_dict(),
                entry_agent_id=entry_id,
                root_invocation_id=current_invocation_id,
            )
            entry_agent = await self._resolve_entry_agent(project_id, entry_id)
            if entry_agent is None:
                raise RuntimeError(f"入口 agent 不可用：{entry_id}")

            # Build the reporter now that we know the entry agent's name.
            reporter = await self._reporter_for_run(
                task.run,
                entry_agent_name=getattr(entry_agent, "name", entry_id),
            )
            outputs_agent_name = getattr(entry_agent, "name", entry_id)

            await run_store.update(task.run.id, status="classifying")
            if not task.is_resume:
                await self._safe_reply(
                    task.inbound.message_id,
                    f"已接单：{getattr(entry_agent, 'name', entry_id)} 处理此任务。\nRun ID: {task.run.id}",
                    f"{task.run.id}-accepted",
                )
            await reporter.start(agent_id=entry_id, phase="entry")

            # --- Run the entry agent (it drives collaboration via MCP delegate) ---
            roster = await orchestrator._build_roster(project_id)
            outputs = await orchestrator.run_entry(
                channel_id=task.binding.channel_id,
                thread_id=task.thread.id,
                content=task_text,
                entry_agent=entry_agent,
                trigger_message_id=task.inbound.message_id,
                run_id=task.run.id,
                on_status=on_status,
                on_progress=reporter.on_progress,
                project_id=project_id,
                roster=roster,
                output_intent=delivery_intent.as_dict(),
                user_id=task.inbound.external_user_id,
                invocation_id=current_invocation_id,
                record_user_message=not task.is_resume,
            )
            outputs["reply_agent_id"] = entry_id
            outputs["reply_agent_name"] = outputs_agent_name
            if outputs.get("suspended"):
                outputs["process"] = reporter.process_snapshot()
                outputs["delivery"] = reporter.delivery_snapshot()
                outputs["progress_event_count"] = reporter.delivery_snapshot()["event_count"]
                await run_store.update(
                    task.run.id,
                    status="waiting_children",
                    outputs=outputs,
                    stage_summary=str(outputs.get("summary") or ""),
                    root_invocation_id=current_invocation_id,
                )
                await reporter.on_progress(
                    "checkpoint",
                    {
                        "agent_id": entry_id,
                        "agent_name": outputs_agent_name,
                        "phase": "waiting",
                        "title": "子任务已在后台接单，等待完成后自动继续",
                    },
                )
                await reporter.pause()
                await self._maybe_resume_parent(task.run.id, current_invocation_id)
                return
            final_message = sanitize_agent_text(
                str(outputs.get("final_message") or outputs.get("summary") or "任务已完成。")
            ).text
            artifacts = list(outputs.get("artifacts") or [])
            final_message, compacted = compact_artifact_final_message(
                final_message,
                artifact_requested=delivery_intent.artifact_requested,
                artifacts=artifacts,
            )
            if compacted:
                outputs["final_message_compacted"] = True
            warning = outputs.get("review_warning")
            if warning:
                final_message = f"{final_message}\n\n注意：{warning}"
            await reporter.on_progress(
                "checkpoint",
                {
                    "agent_id": entry_id,
                    "agent_name": outputs_agent_name,
                    "phase": "summarizing",
                    "title": "分析执行完成，正在整理最终结果",
                },
            )
            # Delivery is decided before execution. Length never grants Wiki
            # write permission and never turns a chat answer into a document.
            if delivery_intent.artifact_requested and not artifacts:
                outputs["archive_status"] = "missing"
                final_message = (
                    f"{final_message}\n\n"
                    "⚠️ 你要求了文档交付，但本次没有成功生成文档。聊天答案仍保留在这里。"
                )
            else:
                outputs["archive_status"] = "artifact" if artifacts else "not_requested"
            outputs["final_message"] = final_message
            outputs["artifacts"] = artifacts
            reporter._card.set_final_message(final_message)
            reporter._card.set_artifacts(artifacts)
            await reporter.on_progress(
                "checkpoint",
                {
                    "agent_id": entry_id,
                    "agent_name": outputs_agent_name,
                    "phase": "summarizing",
                    "title": "最终答案已生成",
                },
            )

            # Finalize first so the persisted diagnostics include the actual
            # final delivery outcome and the closed process stage.
            await run_store.update(task.run.id, status="completing")
            await run_supervisor.stop_run(task.run.id)
            await reporter.finalize()
            outputs["process"] = reporter.process_snapshot()
            outputs["delivery"] = reporter.delivery_snapshot()
            outputs["progress_event_count"] = reporter.delivery_snapshot()["event_count"]
            outputs["run_result"] = RunResult(
                final_message=final_message,
                reply_agent_id=entry_id,
                reply_agent_name=outputs_agent_name,
                artifacts=artifacts,
                process_event_count=int(outputs["progress_event_count"]),
            ).model_dump(mode="json")
            await run_store.update(
                task.run.id,
                status="completed",
                plan={"text": outputs.get("plan", "")},
                outputs=outputs,
            )
            self._reporters.pop(task.run.id, None)
        except RecoverableRunError as exc:
            attempts = (await run_store.get(task.run.id)).recovery_count + 1
            if attempts <= max(settings.recovery_max_attempts, 0):
                await run_store.update(
                    task.run.id,
                    status="recovering",
                    error=sanitize_agent_text(str(exc)).text,
                    recovery_count=attempts,
                    root_invocation_id=current_invocation_id,
                )
                if reporter is not None:
                    reporter._card.set_short_result("当前步骤长时间没有进展，系统将自动恢复任务。")
                    await reporter.pause()
                await self.enqueue_resume(task.run.id, current_invocation_id)
                return
            await run_supervisor.stop_run(task.run.id)
            message = sanitize_agent_text(str(exc)).text
            await run_store.update(task.run.id, status="failed", error=message)
            if reporter is not None:
                reporter._card.set_short_result(f"任务恢复多次仍无进展：{message}")
                await reporter.finalize()
            self._reporters.pop(task.run.id, None)
        except asyncio.CancelledError:
            await run_supervisor.stop_run(task.run.id)
            if reporter is not None:
                await reporter.pause()
            current = await run_store.get(task.run.id)
            await run_store.update(
                task.run.id,
                status="recovering",
                error="服务停止，任务将在重启后从已有成果恢复。",
                recovery_count=(current.recovery_count + 1) if current else 1,
            )
            raise
        except Exception as exc:
            message = sanitize_agent_text(str(exc)).text
            await run_supervisor.stop_run(task.run.id)
            await run_store.update(
                task.run.id,
                status="failed",
                error=message,
            )
            if reporter is not None:
                reporter._card.set_short_result(f"任务执行失败：{message}")
                await reporter.finalize()
            else:
                # Reporter never built (e.g. entry agent missing): text reply.
                await self._safe_reply(
                    task.inbound.message_id,
                    f"任务执行失败：{message}\nRun ID: {task.run.id}",
                    f"{task.run.id}-failed",
                )
            self._reporters.pop(task.run.id, None)

    async def _safe_reply(self, message_id: str, text: str, request_uuid: str) -> None:
        try:
            await self.sender.reply(
                message_id,
                text,
                reply_in_thread=True,
                request_uuid=request_uuid,
            )
        except Exception:
            # The run snapshot still records the result. A failed notification
            # must not crash a worker or cause the task itself to be repeated.
            pass


feishu_task_queue = FeishuTaskQueue()
