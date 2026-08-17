"""Small runtime supervisor for parent/child agent processes.

The supervisor deliberately stays in-memory. Durable task and delegation state
continues to live in Run JSON; after a service restart a fresh process resumes
from the persisted task and workspace instead of trying to reattach pipes.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
import time
from dataclasses import dataclass, field

from config import settings


@dataclass
class InvocationState:
    run_id: str
    invocation_id: str
    parent_invocation_id: str | None = None
    agent_id: str = ""
    process: asyncio.subprocess.Process | None = None
    workspace_dir: str = ""
    last_tool: str = ""
    last_tool_id: str = ""
    last_path: str = ""
    last_path_mtime: float = 0.0
    last_activity_at: float = field(default_factory=time.monotonic)
    last_progress_at: float = field(default_factory=time.monotonic)
    silent_checks: int = 0


class RunSupervisor:
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], InvocationState] = {}
        self._tasks: dict[str, dict[str, asyncio.Task]] = {}
        self._accepting: dict[str, bool] = {}
        self._lock = asyncio.Lock()

    async def register_process(
        self,
        *,
        run_id: str,
        invocation_id: str,
        process: asyncio.subprocess.Process,
        workspace_dir: str = "",
        parent_invocation_id: str | None = None,
        agent_id: str = "",
    ) -> None:
        async with self._lock:
            self._accepting.setdefault(run_id, True)
            self._states[(run_id, invocation_id)] = InvocationState(
                run_id=run_id,
                invocation_id=invocation_id,
                parent_invocation_id=parent_invocation_id,
                agent_id=agent_id,
                process=process,
                workspace_dir=workspace_dir,
            )

    async def unregister_process(self, run_id: str, invocation_id: str) -> None:
        async with self._lock:
            self._states.pop((run_id, invocation_id), None)

    async def register_task(self, run_id: str, task_id: str, task: asyncio.Task) -> None:
        async with self._lock:
            self._accepting.setdefault(run_id, True)
            self._tasks.setdefault(run_id, {})[task_id] = task

    async def unregister_task(self, run_id: str, task_id: str) -> None:
        async with self._lock:
            tasks = self._tasks.get(run_id)
            if tasks:
                tasks.pop(task_id, None)

    async def accepting(self, run_id: str) -> bool:
        async with self._lock:
            return self._accepting.get(run_id, True)

    async def record_activity(self, run_id: str, invocation_id: str) -> None:
        async with self._lock:
            state = self._states.get((run_id, invocation_id))
            if state:
                state.last_activity_at = time.monotonic()

    async def record_tool_call(
        self,
        run_id: str,
        invocation_id: str,
        tool: str,
        tool_id: str = "",
        args: dict | None = None,
    ) -> None:
        path = ""
        if isinstance(args, dict):
            path = str(args.get("file_path") or args.get("path") or "")
        async with self._lock:
            state = self._states.get((run_id, invocation_id))
            if not state:
                return
            state.last_activity_at = time.monotonic()
            state.last_tool = tool[:100]
            state.last_tool_id = tool_id[:100]
            if path:
                state.last_path = path[:1000]
                try:
                    state.last_path_mtime = Path(path).stat().st_mtime
                except OSError:
                    state.last_path_mtime = 0.0

    async def record_progress(self, run_id: str, invocation_id: str) -> None:
        async with self._lock:
            state = self._states.get((run_id, invocation_id))
            if state:
                now = time.monotonic()
                state.last_activity_at = now
                state.last_progress_at = now
                state.silent_checks = 0

    async def assess(self, run_id: str, invocation_id: str) -> str:
        async with self._lock:
            state = self._states.get((run_id, invocation_id))
            if state is None:
                return "working"
            if not self._accepting.get(run_id, True):
                return "cancelled"
            proc = state.process
            if proc is not None and proc.returncode is not None:
                return "exited"

            now = time.monotonic()
            threshold = max(float(settings.stall_check_after), 1.0)
            if now - state.last_progress_at < threshold:
                return "working"

            # A recently active stdout/tool stream is evidence that the current
            # long step is still alive, even if it has not produced an artifact.
            if now - state.last_activity_at < threshold:
                return "working"

            if state.last_path:
                try:
                    mtime = Path(state.last_path).stat().st_mtime
                except OSError:
                    mtime = state.last_path_mtime
                if mtime > state.last_path_mtime:
                    state.last_path_mtime = mtime
                    state.last_progress_at = now
                    state.last_activity_at = now
                    state.silent_checks = 0
                    return "working"

            state.silent_checks += 1
            if state.silent_checks < max(int(settings.stall_confirmations), 1):
                return "suspected_stall"
            return "stalled"

    async def stop_run(self, run_id: str) -> None:
        async with self._lock:
            self._accepting[run_id] = False
            states = [s for (rid, _), s in self._states.items() if rid == run_id]
            tasks = list(self._tasks.get(run_id, {}).values())

        current = asyncio.current_task()
        for task in tasks:
            if task is not current and not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(
                *(task for task in tasks if task is not current),
                return_exceptions=True,
            )

        for state in states:
            proc = state.process
            if proc is None or proc.returncode is not None:
                continue
            if os.name == "nt" and getattr(proc, "pid", None):
                try:
                    await asyncio.to_thread(
                        subprocess.run,
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=5,
                    )
                except Exception:
                    pass
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass

        async with self._lock:
            self._states = {
                key: value for key, value in self._states.items() if key[0] != run_id
            }
            self._tasks.pop(run_id, None)

    async def reopen_run(self, run_id: str) -> None:
        async with self._lock:
            self._accepting[run_id] = True


run_supervisor = RunSupervisor()
