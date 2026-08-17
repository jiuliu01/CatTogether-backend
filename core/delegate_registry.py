"""Delegation context registry: maps a per-run token to the context an agent
needs to delegate (project, channel, thread, depth, run_id).

Each agent subprocess is issued a token via .mcp.json. When that agent calls
`delegate`, the MCP server posts the token to /internal/delegate; the backend
looks it up here to recover the context, checks depth/concurrency, and runs
the target agent under a fresh (depth+1) token.

Concurrency across all active delegations is bounded by a semaphore
(CT_DELEGATE_MAX_CONCURRENCY); recursion depth by CT_DELEGATE_MAX_DEPTH.
"""
from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import uuid


ProgressCallback = Callable[[str, dict], Awaitable[None]]


@dataclass
class DelegateContext:
    token: str
    project_id: str | None
    channel_id: str
    thread_id: str | None
    run_id: str | None
    depth: int
    caller_agent_id: str | None = None  # who holds this token (to block self-delegation)
    caller_agent_name: str = ""
    phase: str = "entry"
    on_progress: ProgressCallback | None = None
    created_at: float = 0.0
    invocation_id: str = ""
    parent_invocation_id: str | None = None
    mcp_capabilities: tuple[str, ...] = ()
    output_intent: dict | None = None
    artifacts: list[dict] | None = None
    user_id: str = "default"


class DelegateContextRegistry:
    def __init__(self) -> None:
        self._contexts: dict[str, DelegateContext] = {}
        self._lock = asyncio.Lock()
        from config import settings
        self._semaphore = asyncio.Semaphore(max(settings.delegate_max_concurrency, 1))

    @property
    def max_depth(self) -> int:
        from config import settings
        return max(int(settings.delegate_max_depth), 1)

    async def issue(
        self,
        *,
        project_id: str | None,
        channel_id: str,
        thread_id: str | None,
        run_id: str | None,
        depth: int = 0,
        caller_agent_id: str | None = None,
        caller_agent_name: str = "",
        phase: str = "entry",
        on_progress: ProgressCallback | None = None,
        invocation_id: str | None = None,
        parent_invocation_id: str | None = None,
        mcp_capabilities: list[str] | tuple[str, ...] | None = None,
        output_intent: dict | None = None,
        artifacts: list[dict] | None = None,
        user_id: str = "default",
    ) -> str:
        token = secrets.token_urlsafe(24)
        ctx = DelegateContext(
            token=token,
            project_id=project_id,
            channel_id=channel_id,
            thread_id=thread_id,
            run_id=run_id,
            depth=depth,
            caller_agent_id=caller_agent_id,
            caller_agent_name=caller_agent_name,
            phase=phase,
            on_progress=on_progress,
            created_at=time.monotonic(),
            invocation_id=invocation_id or uuid.uuid4().hex,
            parent_invocation_id=parent_invocation_id,
            mcp_capabilities=tuple(
                ("delegate", "report_progress")
                if mcp_capabilities is None
                else mcp_capabilities
            ),
            output_intent=dict(output_intent or {}),
            artifacts=artifacts if artifacts is not None else [],
            user_id=user_id,
        )
        async with self._lock:
            self._contexts[token] = ctx
        return token

    async def get(self, token: str) -> DelegateContext | None:
        async with self._lock:
            return self._contexts.get(token)

    async def revoke(self, token: str) -> None:
        async with self._lock:
            self._contexts.pop(token, None)

    def acquire(self) -> asyncio.Semaphore:
        """Return the concurrency semaphore callers should `async with`."""
        return self._semaphore


delegate_registry = DelegateContextRegistry()
