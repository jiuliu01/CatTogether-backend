"""Base for CLI agents (Codex / Claude Code).

Spawns a short-lived subprocess per invoke, streams stdout, and parses chunks
via subclass parse_chunk. Auth is API-key via environment variables inherited
by the child process — never interactive login.

Lifecycle: one process per invoke. No daemon, no heartbeat polling (simplified
from openagents' BaseAdapter).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from typing import AsyncIterator

from agents.base import BaseAgent, InvokeContext, RawEvent
from config import settings


class CLIBaseAgent(BaseAgent):
    def __init__(self, agent_id: str, name: str, bin_path: str | None, description: str = "") -> None:
        super().__init__(agent_id, name, "cli", description)
        self.bin_path = bin_path  # explicit path or None (resolve via PATH)

    def resolve_bin(self) -> str | None:
        if self.bin_path and os.path.isfile(self.bin_path):
            return self.bin_path
        return shutil.which(self._bin_name())

    def _bin_name(self) -> str:
        return self.name.lower()

    async def health(self) -> bool:
        binary = self.resolve_bin()
        if not binary:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                "--version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5)
            return proc.returncode == 0
        except Exception:
            return False

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        # API keys are inherited from the backend env; ensure they propagate.
        if settings.openai_api_key:
            env.setdefault("OPENAI_API_KEY", settings.openai_api_key)
        if settings.anthropic_api_key:
            env.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
        # Some Anthropic-compatible proxies (and Claude Code itself) auth via
        # ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL rather than ANTHROPIC_API_KEY.
        # Propagate both so the child CLI can reach the same endpoint the host
        # is configured against. ``setdefault`` keeps any explicit child value.
        for var in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
            val = os.environ.get(var)
            if val:
                env.setdefault(var, val)
        return env

    # subclass hooks
    def build_command(self, ctx: InvokeContext) -> list[str]:
        raise NotImplementedError

    def build_prompt(self, ctx: InvokeContext) -> str:
        parts = []
        if ctx.memory:
            parts.append(ctx.memory)
        parts.append(f"User message:\n{ctx.user_message}")
        return "\n\n---\n\n".join(parts)

    async def parse_chunk(self, raw: str, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        """Parse one stdout chunk. Default: pass through as text_delta."""
        if raw.strip():
            yield ("text_delta", {"delta": raw})

    @staticmethod
    async def _terminate_process_tree(proc: asyncio.subprocess.Process) -> None:
        """Stop the CLI and its MCP children, not only the top-level process."""
        if proc.returncode is not None:
            return
        if os.name == "nt" and getattr(proc, "pid", None):
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        subprocess.run,
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=5,
                    ),
                    timeout=6,
                )
            except Exception:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            if proc.returncode is None:
                try:
                    # Also notify asyncio's subprocess transport so its pipe
                    # handles are closed even when taskkill already ended it.
                    proc.kill()
                except ProcessLookupError:
                    pass
        else:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass

    async def invoke(self, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        bin = self.resolve_bin()
        if not bin:
            yield ("error", {"message": f"{self._bin_name()} CLI not found"})
            return

        cmd = self.build_command(ctx)
        prompt = self.build_prompt(ctx)
        env = self._build_env()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=settings.cli_stream_limit,
                cwd=ctx.workspace_dir or None,
                env=env,
            )
        except Exception as e:
            yield ("error", {"message": f"failed to start {self._bin_name()}: {e}"})
            return

        assert proc.stdin and proc.stdout and proc.stderr

        managed_run = bool(ctx.root_run_id and ctx.invocation_id)
        if managed_run:
            from core.run_supervisor import run_supervisor
            await run_supervisor.register_process(
                run_id=str(ctx.root_run_id),
                invocation_id=str(ctx.invocation_id),
                parent_invocation_id=ctx.parent_invocation_id,
                agent_id=self.agent_id,
                process=proc,
                workspace_dir=ctx.workspace_dir,
            )
            if not await run_supervisor.accepting(str(ctx.root_run_id)):
                await self._terminate_process_tree(proc)
                await run_supervisor.unregister_process(
                    str(ctx.root_run_id), str(ctx.invocation_id)
                )
                yield ("status", {"state": "idle"})
                return

        # feed prompt to stdin
        try:
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
        except Exception:
            pass
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.stdin.wait_closed(), timeout=1)
        except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError):
            pass

        cancel = ctx.cancel_event
        spec_timeout = getattr(ctx.spec, "timeout", None) if ctx.spec else None
        timeout = ctx.timeout if ctx.timeout is not None else (spec_timeout if spec_timeout is not None else settings.cli_timeout)
        # Managed CatTogether Runs have no fixed wall-clock death time. They
        # are polled for activity/progress by RunSupervisor instead. Explicit
        # standalone invocations keep the old timeout behavior for callers
        # that intentionally request a bounded command.
        deadline = (
            None
            if managed_run
            else (time.monotonic() + timeout if timeout and timeout > 0 else None)
        )
        probe_interval = max(float(settings.runtime_probe_interval), 0.05)

        async def read_stderr() -> str:
            return (await proc.stderr.read()).decode("utf-8", errors="replace")

        stderr_task = asyncio.create_task(read_stderr())
        stdout_tail: list[str] = []
        last_tool_name = ""
        last_tool_id = ""
        last_persisted_activity = 0.0

        async def timeout_message() -> str:
            await self._terminate_process_tree(proc)
            try:
                stderr = await asyncio.wait_for(asyncio.shield(stderr_task), timeout=1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                stderr = ""
            from core.output_safety import sanitize_agent_text

            parts = [f"{self._bin_name()} 超时（{timeout}s）"]
            if last_tool_name:
                tool = f"{last_tool_name} ({last_tool_id})" if last_tool_id else last_tool_name
                parts.append(f"最后工具：{tool}")
            diagnostic = sanitize_agent_text(stderr.strip()).text[:300]
            if diagnostic:
                parts.append(f"stderr：{diagnostic}")
            return "；".join(parts)

        try:
            while True:
                if cancel and cancel.is_set():
                    await self._terminate_process_tree(proc)
                    yield ("status", {"state": "idle"})
                    return
                if managed_run:
                    try:
                        line = await asyncio.wait_for(
                            proc.stdout.readline(), timeout=probe_interval
                        )
                    except asyncio.TimeoutError:
                        from core.run_supervisor import run_supervisor
                        assessment = await run_supervisor.assess(
                            str(ctx.root_run_id), str(ctx.invocation_id)
                        )
                        if assessment in ("working", "suspected_stall"):
                            continue
                        if assessment == "cancelled":
                            await self._terminate_process_tree(proc)
                            yield ("status", {"state": "idle"})
                            return
                        if assessment == "exited":
                            break
                        await self._terminate_process_tree(proc)
                        yield (
                            "recoverable_error",
                            {
                                "message": (
                                    "Agent 长时间没有工具结果、输出或文件变化，"
                                    "已停止本次执行，准备从现有成果恢复。"
                                )
                            },
                        )
                        return
                elif deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        yield (
                            "error",
                            {"message": await timeout_message()},
                        )
                        return
                    try:
                        line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                    except asyncio.TimeoutError:
                        yield (
                            "error",
                            {"message": await timeout_message()},
                        )
                        return
                else:
                    line = await proc.stdout.readline()
                if not line:
                    break
                raw = line.decode("utf-8", errors="replace")
                if managed_run:
                    from core.run_supervisor import run_supervisor
                    await run_supervisor.record_activity(
                        str(ctx.root_run_id), str(ctx.invocation_id)
                    )
                    now = time.monotonic()
                    if now - last_persisted_activity >= 5:
                        from core.run_store import run_store
                        await run_store.record_runtime_progress(
                            str(ctx.root_run_id), activity=True
                        )
                        last_persisted_activity = now
                stdout_tail.append(raw.strip())
                stdout_tail = stdout_tail[-3:]
                async for ev in self.parse_chunk(raw, ctx):
                    if ev[0] == "tool_call":
                        last_tool_name = str(ev[1].get("tool") or "")[:100]
                        last_tool_id = str(ev[1].get("tool_use_id") or "")[:100]
                        if managed_run:
                            from core.run_supervisor import run_supervisor
                            await run_supervisor.record_tool_call(
                                str(ctx.root_run_id),
                                str(ctx.invocation_id),
                                last_tool_name,
                                last_tool_id,
                                ev[1].get("args") if isinstance(ev[1].get("args"), dict) else {},
                            )
                            from core.run_store import run_store
                            args = ev[1].get("args") if isinstance(ev[1].get("args"), dict) else {}
                            changed_path = str(args.get("file_path") or args.get("path") or "")
                            await run_store.record_runtime_progress(
                                str(ctx.root_run_id),
                                activity=True,
                                tool=last_tool_name,
                                tool_id=last_tool_id,
                                changed_path=changed_path or None,
                            )
                    elif ev[0] == "tool_result" and managed_run:
                        from core.run_supervisor import run_supervisor
                        await run_supervisor.record_progress(
                            str(ctx.root_run_id), str(ctx.invocation_id)
                        )
                        from core.run_store import run_store
                        await run_store.record_runtime_progress(
                            str(ctx.root_run_id), activity=True, progress=True
                        )
                    yield ev

            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                await self._terminate_process_tree(proc)
            err = await stderr_task
            rc = proc.returncode
            if rc not in (0, None):
                detail = err.strip()
                if not detail:
                    detail = " | ".join(line for line in stdout_tail if line)
                yield (
                    "error",
                    {
                        "message": (
                            f"{self._bin_name()} exited {rc}: "
                            f"{detail[:1000] or 'no diagnostic output'}"
                        )
                    },
                )
                return
        except Exception as e:
            yield ("error", {"message": str(e)})
            return
        finally:
            if not stderr_task.done():
                stderr_task.cancel()
            if proc.returncode is None:
                await self._terminate_process_tree(proc)
            if managed_run:
                from core.run_supervisor import run_supervisor
                await run_supervisor.unregister_process(
                    str(ctx.root_run_id), str(ctx.invocation_id)
                )

        yield ("done", {})
