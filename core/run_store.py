"""JSON snapshots for multi-agent coordinator runs."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings
from models.schemas import AgentRun, DelegationRecord


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunStore:
    def __init__(self) -> None:
        self._dir = settings.data_dir / "feishu" / "runs"
        self._runs: dict[str, AgentRun] = {}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        for path in self._dir.glob("*.json"):
            try:
                run = AgentRun.model_validate_json(path.read_text(encoding="utf-8"))
                self._runs[run.id] = run
            except (OSError, ValueError):
                continue

    def _write(self, run: AgentRun) -> None:
        path = self._dir / f"{run.id}.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        temp.replace(path)

    async def create(
        self,
        channel_id: str,
        thread_id: str,
        trigger_message_id: str,
        trigger_user_id: str | None = None,
    ) -> AgentRun:
        run = AgentRun(
            id=uuid.uuid4().hex,
            channel_id=channel_id,
            thread_id=thread_id,
            trigger_message_id=trigger_message_id,
            trigger_user_id=trigger_user_id,
        )
        async with self._lock:
            self._runs[run.id] = run
            self._write(run)
        return run

    async def update(
        self,
        run_id: str,
        *,
        status: str | None = None,
        plan: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        intent: dict[str, Any] | None = None,
        error: str | None = None,
        original_task: str | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        entry_agent_id: str | None = None,
        root_invocation_id: str | None = None,
        tenant_key: str | None = None,
        chat_id: str | None = None,
        sender_open_id: str | None = None,
        recovery_count: int | None = None,
        stage_summary: str | None = None,
    ) -> AgentRun | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            terminal = {"completed", "failed", "cancelled"}
            if run.status in terminal:
                return run
            if status is not None and not (
                run.status in terminal and status != run.status
            ):
                run.status = status  # type: ignore[assignment]
            if plan is not None:
                run.plan = plan
            if outputs is not None:
                # Delivery/card identity and prior stage diagnostics must
                # survive status updates and parent-agent resumptions.
                run.outputs = {**run.outputs, **outputs}
            if intent is not None:
                run.intent = intent
            if error is not None:
                run.error = error
            for key, value in {
                "original_task": original_task,
                "project_id": project_id,
                "workspace_dir": workspace_dir,
                "entry_agent_id": entry_agent_id,
                "root_invocation_id": root_invocation_id,
                "tenant_key": tenant_key,
                "chat_id": chat_id,
                "sender_open_id": sender_open_id,
                "recovery_count": recovery_count,
                "stage_summary": stage_summary,
            }.items():
                if value is not None:
                    setattr(run, key, value)
            run.updated_at = _now()
            self._write(run)
            return run

    async def get(self, run_id: str) -> AgentRun | None:
        return self._runs.get(run_id)

    async def patch_outputs(self, run_id: str, values: dict[str, Any]) -> AgentRun | None:
        """Merge diagnostic/progress fields without replacing existing outputs."""
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            if run.status in ("completed", "failed", "cancelled"):
                return run
            run.outputs = {**run.outputs, **values}
            run.updated_at = _now()
            self._write(run)
            return run

    async def list(self, limit: int = 100) -> list[AgentRun]:
        return sorted(
            self._runs.values(),
            key=lambda run: run.created_at,
            reverse=True,
        )[:limit]

    @staticmethod
    def _refresh_pending(run: AgentRun) -> None:
        run.pending_child_ids = [
            item.id
            for item in run.delegations
            if not item.consumed
            and item.status not in ("completed", "failed", "cancelled")
        ]

    async def add_delegation(
        self, run_id: str, delegation: DelegationRecord
    ) -> AgentRun | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run or run.status in ("completed", "failed", "cancelled", "completing"):
                return None
            if any(item.id == delegation.id for item in run.delegations):
                return run
            run.delegations.append(delegation)
            self._refresh_pending(run)
            run.updated_at = _now()
            self._write(run)
            return run

    async def get_delegation(
        self, run_id: str, child_id: str
    ) -> DelegationRecord | None:
        run = self._runs.get(run_id)
        if not run:
            return None
        return next((item for item in run.delegations if item.id == child_id), None)

    async def update_delegation(
        self,
        run_id: str,
        child_id: str,
        *,
        status: str | None = None,
        result: str | None = None,
        error: str | None = None,
        recovery_count: int | None = None,
        consumed: bool | None = None,
    ) -> DelegationRecord | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            if run.status in ("completed", "failed", "cancelled"):
                return None
            child = next((item for item in run.delegations if item.id == child_id), None)
            if child is None:
                return None
            if status is not None:
                child.status = status  # type: ignore[assignment]
            if result is not None:
                child.result = result
            if error is not None:
                child.error = error
            if recovery_count is not None:
                child.recovery_count = recovery_count
            if consumed is not None:
                child.consumed = consumed
            child.updated_at = _now()
            self._refresh_pending(run)
            run.last_progress_at = _now()
            run.updated_at = _now()
            self._write(run)
            return child

    async def children_for_parent(
        self,
        run_id: str,
        parent_invocation_id: str,
        *,
        include_consumed: bool = False,
    ) -> list[DelegationRecord]:
        run = self._runs.get(run_id)
        if not run:
            return []
        return [
            item for item in run.delegations
            if item.parent_invocation_id == parent_invocation_id
            and (include_consumed or not item.consumed)
        ]

    async def mark_children_consumed(
        self, run_id: str, parent_invocation_id: str
    ) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return
            if run.status in ("completed", "failed", "cancelled"):
                return
            for item in run.delegations:
                if item.parent_invocation_id == parent_invocation_id:
                    item.consumed = True
                    item.updated_at = _now()
            if parent_invocation_id in run.resume_queued_for:
                run.resume_queued_for.remove(parent_invocation_id)
            self._refresh_pending(run)
            run.updated_at = _now()
            self._write(run)

    async def try_mark_resume_queued(
        self, run_id: str, parent_invocation_id: str
    ) -> bool:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run or run.status in ("completed", "failed", "cancelled", "completing"):
                return False
            children = [
                item for item in run.delegations
                if item.parent_invocation_id == parent_invocation_id and not item.consumed
            ]
            if not children or any(
                item.status not in ("completed", "failed", "cancelled")
                for item in children
            ):
                return False
            if parent_invocation_id in run.resume_queued_for:
                return False
            run.resume_queued_for.append(parent_invocation_id)
            run.updated_at = _now()
            self._write(run)
            return True

    async def clear_resume_queued(
        self, run_id: str, parent_invocation_id: str
    ) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return
            if run.status in ("completed", "failed", "cancelled"):
                return
            if parent_invocation_id in run.resume_queued_for:
                run.resume_queued_for.remove(parent_invocation_id)
                run.updated_at = _now()
                self._write(run)

    async def record_runtime_progress(
        self,
        run_id: str,
        *,
        activity: bool = False,
        progress: bool = False,
        tool: str | None = None,
        tool_id: str | None = None,
        changed_path: str | None = None,
    ) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return
            if run.status in ("completed", "failed", "cancelled"):
                return
            now = _now()
            if activity:
                run.last_activity_at = now
            if progress:
                run.last_progress_at = now
            if tool is not None:
                run.last_tool = tool[:100]
            if tool_id is not None:
                run.last_tool_id = tool_id[:100]
            if changed_path and changed_path not in run.last_changed_files:
                run.last_changed_files = (run.last_changed_files + [changed_path])[-100:]
            run.updated_at = now
            self._write(run)


run_store = RunStore()
