"""Orchestrator: decides how a user message is dispatched to one or more agents.

Strategies:
  - direct:      the mentioned agent (or the single channel agent) replies.
  - round_robin: every agent in the channel replies in sequence.
  - fan_out:     every agent replies in parallel; results merged.

Flow per agent:
  history -> memory -> InvokeContext -> agent.invoke -> publish events to bus
  -> on done, append final message to session -> update long-term memory.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Literal
from collections.abc import Awaitable, Callable

from agents.base import BaseAgent, InvokeContext
from agents.roles import ROLE_SYSTEM_GUIDANCE, resolve_role_agents
from config import settings
from core.event_bus import event_bus
from core.output_safety import sanitize_agent_text
from core.registry import registry
from core.session_manager import session_manager
from core.workspace import workspace_manager
from core.workspace_changes import changed_workspace_paths, snapshot_workspace


logger = logging.getLogger(__name__)

def _uuid() -> str:
    return uuid.uuid4().hex


def _public_event_data(event_type: str, data: dict) -> dict:
    """Keep public/WebSocket events useful without exposing raw tool payloads."""
    if event_type == "tool_result":
        return {
            "tool": sanitize_agent_text(str(data.get("tool") or "tool")).text[:100],
            "result": "工具已完成；原始结果不直接展示。",
        }
    if event_type == "tool_call":
        args = data.get("args") if isinstance(data.get("args"), dict) else {}
        safe_args = {}
        for key in ("target", "access", "file_path", "path", "pattern", "query", "task", "title"):
            if key in args:
                safe_args[key] = sanitize_agent_text(str(args[key])).text[:300]
        return {
            "tool": sanitize_agent_text(str(data.get("tool") or "tool")).text[:100],
            "args": safe_args,
            "tool_use_id": str(data.get("tool_use_id") or "")[:100],
        }
    safe = dict(data)
    for key in ("delta", "text", "message", "detail"):
        if isinstance(safe.get(key), str):
            safe[key] = sanitize_agent_text(safe[key]).text
    return safe


def _safe_final_message(text: str) -> str:
    safe = sanitize_agent_text(text)
    value = safe.text
    if safe.findings and "检测到敏感凭据样式" not in value:
        value = (
            f"{value}\n\n"
            "⚠️ 系统检测到敏感凭据样式并已隐藏；如果原内容是真实凭据，请立即轮换。"
        ).strip()
    return value


def _agent_role(agent: BaseAgent) -> str:
    spec = getattr(agent, "_spec", None)
    role = getattr(spec, "role", None)
    if isinstance(role, str) and role.strip():
        return role.strip().lower()
    return agent.agent_id.lower() if agent.agent_id else "custom"


def _memory_role(agent_role: str) -> str:
    """Map orchestrator agent roles to Memory 2.1 roles.

    Only ``coordinator`` maps directly; all other agent roles (research,
    coding, reviewer, custom, …) map to ``agent``.
    """
    if agent_role == "coordinator":
        return "coordinator"
    return "agent"


def _memory_actor(
    agent: BaseAgent | None = None,
    *,
    user_id: str = "default",
    agent_role: str | None = None,
) -> "ActorContext":
    """Build a Memory 2.1 ActorContext for the current invocation."""
    from memory.permissions import ActorContext

    if agent is not None:
        role = _memory_role(_agent_role(agent))
        actor_id = agent.agent_id
    else:
        role = "user"
        actor_id = user_id
    from memory.scope import tenant_id_from_user_id
    return ActorContext(
        tenant_id=tenant_id_from_user_id(user_id),
        role=role,  # type: ignore[arg-type]
        actor_id=actor_id,
        display_name=actor_id,
    )


def _mutation_path(event_type: str, data: dict, pending: dict[str, str]) -> str | None:
    if event_type == "tool_call" and str(data.get("tool") or "").lower() in {
        "edit", "write", "file_change",
    }:
        args = data.get("args") if isinstance(data.get("args"), dict) else {}
        value = args.get("file_path") or args.get("path")
        key = str(data.get("tool_use_id") or value or "")
        if key:
            pending[key] = str(value)[:500] if value else "[workspace change]"
        return None
    if event_type == "tool_result":
        tool = str(data.get("tool") or "")
        if tool in pending:
            return pending.pop(tool)
        if tool.lower() != "file_change":
            return None
        result = data.get("result")
        if isinstance(result, dict):
            value = result.get("path") or result.get("file_path")
            return str(value)[:500] if value else "[workspace change]"
        return "[workspace change]"
    return None


def _merge_changed_paths(reported: list[str], detected: list[str]) -> list[str]:
    return list(dict.fromkeys(path for path in [*reported, *detected] if path))


async def _submit_invocation_memory(
    *,
    user_id: str,
    project_id: str | None,
    channel_id: str,
    thread_id: str | None,
    run_id: str | None,
    agent: BaseAgent,
    user_text: str,
    final_text: str,
    mutation_paths: list[str],
    error: str | None,
) -> None:
    """Submit durable state even when an invocation left partial file changes."""
    succeeded = not bool(error)

    # ----- Memory 2.2: LLM extraction → Qdrant -----
    # When the v22 switch is on, the memory agent distils this invocation into
    # flat memories in Qdrant. This is the primary write path for 2.2; the
    # v2.1 rule-based write_propose below still runs (for the session_messages
    # history append and as a fallback) unless we later choose to retire it.
    from config import settings
    if settings.memory_v22_enabled:
        try:
            from memory.v22 import get_memory_extractor
            from memory.v22.extractor import ExtractionContext
            extractor = get_memory_extractor()
            if extractor is not None:
                await extractor.extract(ExtractionContext(
                    event_type="agent_invocation",
                    agent_id=agent.agent_id,
                    agent_name=agent.name,
                    user_text=user_text,
                    final_text=final_text,
                    mutation_paths=list(mutation_paths),
                    succeeded=succeeded,
                    run_id=run_id,
                    channel_id=channel_id,
                    thread_id=thread_id,
                    project_id=project_id,
                    user_id=user_id,
                ))
        except Exception:
            # v22 extraction is a non-fatal side effect.
            logger.exception("v22 memory extraction failed (non-fatal)")

    # ----- Memory 2.1: rule-based write_propose (kept for history append + fallback) -----
    from memory.memory_api import MemoryAPI, WriteProposal
    from memory.permissions import ActorContext

    allowed_domains: list[str] = []
    if project_id and (succeeded or mutation_paths):
        allowed_domains.append("project")
    if succeeded and final_text:
        allowed_domains.append("agent")
    if not allowed_domains:
        return

    actor = _memory_actor(agent, user_id=user_id)
    api = MemoryAPI()
    proposal = WriteProposal(
        text=final_text or user_text,
        domain="project" if "project" in allowed_domains else "agent",
        scope_hint=project_id or "default",
        kind="project_fact",
        importance=0.5,
        confidence=0.5,
        source_type="agent_result",
        source_ids=[run_id] if run_id else [],
        actor_id=agent.agent_id,
        event_type="agent_invocation",
        user_text=user_text,
        final_text=final_text,
        mutation_paths=mutation_paths,
        allowed_domains=allowed_domains,
        succeeded=succeeded,
        channel_id=channel_id,
        thread_id=thread_id,
        run_id=run_id,
        agent_role=_agent_role(agent),
        workspace_id=project_id,
    )
    try:
        await api.write_propose(proposal, actor=actor)
    except Exception:
        # Memory write failures must never break the orchestrator flow.
        pass
    if final_text:
        history_id = f"sm_{uuid.uuid5(uuid.NAMESPACE_URL, f'{run_id}:{agent.agent_id}:{final_text}').hex[:16]}"
        try:
            await api.history_append(
                channel_id,
                "agent",
                final_text,
                actor=actor,
                thread_id=thread_id,
                agent_id=agent.agent_id,
                message_id=history_id,
            )
        except Exception:
            pass


class Orchestrator:
    def __init__(self) -> None:
        # channel_id -> running task
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel: dict[str, asyncio.Event] = {}
        self._thread_tasks: dict[str, asyncio.Task] = {}
        self._thread_cancel: dict[str, asyncio.Event] = {}

    @staticmethod
    async def _build_roster(project_id: str | None) -> str:
        """Build the 同事名单 string injected into an agent's prompt."""
        from core.agent_templates import roster_block
        project_agent_ids = None
        if project_id:
            try:
                from core.agent_store import agent_store
                specs = await agent_store.list(project_id)
                # Exclude internal agents (e.g. memory extractor) — they are
                # not delegate targets and must not appear in the roster.
                project_agent_ids = [
                    s.agent_id for s in specs if not getattr(s, "internal", False)
                ]
            except Exception:
                project_agent_ids = None
        return roster_block(project_agent_ids)

    def is_running(self, channel_id: str) -> bool:
        t = self._tasks.get(channel_id)
        return t is not None and not t.done()

    def cancel(self, channel_id: str) -> bool:
        ev = self._cancel.get(channel_id)
        if ev:
            ev.set()
            return True
        return False

    async def dispatch(
        self,
        channel_id: str,
        content: str,
        mentioned: list[str] | None,
        strategy: str | None,
        user_id: str = "default",
    ) -> None:
        """Run dispatch as a background task (non-blocking)."""
        if self.is_running(channel_id):
            raise RuntimeError("a dispatch is already running on this channel")
        cancel_ev = asyncio.Event()
        self._cancel[channel_id] = cancel_ev
        task = asyncio.create_task(
            self._run(channel_id, content, mentioned or [], strategy or "direct", user_id)
        )
        self._tasks[channel_id] = task
        try:
            await task
        except Exception as e:
            await event_bus.publish(channel_id, "system", "error", {"message": str(e)})
        finally:
            self._tasks.pop(channel_id, None)
            self._cancel.pop(channel_id, None)

    async def _run(
        self,
        channel_id: str,
        content: str,
        mentioned: list[str],
        strategy: str,
        user_id: str = "default",
    ) -> None:
        # persist user message
        await session_manager.append_message(channel_id, "user", content)

        channel_agents = await session_manager.get_agents(channel_id)
        # resolve target agents
        targets: list[BaseAgent] = []
        if mentioned:
            for aid in mentioned:
                a = registry.get(aid)
                if a and a not in targets:
                    targets.append(a)
        if not targets:
            for aid in channel_agents:
                a = registry.get(aid)
                if a and a not in targets:
                    targets.append(a)
        if not targets:
            await event_bus.publish(channel_id, "system", "error", {"message": "no agents available"})
            return

        # effective strategy
        eff: Literal["direct", "round_robin", "fan_out"] = "direct"  # type: ignore[assignment]
        if strategy in ("direct", "round_robin", "fan_out"):
            eff = strategy  # type: ignore[assignment]
        if len(targets) == 1:
            eff = "direct"

        cancel_ev = self._cancel.get(channel_id)

        results: list[tuple[str, str | None]] = []
        if eff == "fan_out":
            results = list(await asyncio.gather(*[
                self._run_one(channel_id, a, content, cancel_ev, user_id=user_id)
                for a in targets
            ]))
        else:  # direct or round_robin: sequential
            for a in targets:
                if cancel_ev and cancel_ev.is_set():
                    break
                results.append(await self._run_one(
                    channel_id, a, content, cancel_ev, user_id=user_id
                ))

        from memory.memory_api import MemoryAPI, WriteProposal
        final_text = "\n".join(text for text, error in results if text and not error)
        if final_text:
            # Memory 2.2: conversation_completed trigger.
            if settings.memory_v22_enabled:
                try:
                    from memory.v22 import get_memory_extractor
                    from memory.v22.extractor import ExtractionContext
                    extractor = get_memory_extractor()
                    if extractor is not None:
                        await extractor.extract(ExtractionContext(
                            event_type="conversation_completed",
                            agent_id=targets[0].agent_id,
                            agent_name=targets[0].name,
                            user_text=content,
                            final_text=final_text,
                            mutation_paths=[],
                            succeeded=True,
                            channel_id=channel_id,
                            user_id=user_id,
                        ))
                except Exception:
                    logger.exception("v22 memory extraction failed (non-fatal)")

            actor = _memory_actor(targets[0], user_id=user_id)
            api = MemoryAPI()
            proposal = WriteProposal(
                text=final_text,
                domain="user",
                scope_hint=user_id,
                kind="project_fact",
                importance=0.5,
                confidence=0.5,
                source_type="agent_result",
                actor_id=targets[0].agent_id,
                event_type="conversation_completed",
                user_text=content,
                final_text=final_text,
                allowed_domains=["user"],
                channel_id=channel_id,
                agent_role=_agent_role(targets[0]),
            )
            try:
                await api.write_propose(proposal, actor=actor)
            except Exception:
                pass

        await event_bus.publish(channel_id, "system", "done", {"message": "dispatch complete"})

    async def _run_one(
        self,
        channel_id: str,
        agent: BaseAgent,
        content: str,
        cancel_ev: asyncio.Event | None,
        *,
        user_id: str = "default",
    ) -> tuple[str, str | None]:
        return await self._invoke_collect(
            channel_id, agent, content, cancel_ev, user_id=user_id
        )

    async def invoke_for_delegate(
        self,
        *,
        agent: BaseAgent,
        content: str,
        channel_id: str,
        thread_id: str | None,
        project_id: str | None,
        access: str,
        delegate_token: str,
        mcp_capabilities: list[str] | None = None,
        output_intent: dict | None = None,
        on_progress: Callable[[str, dict], Awaitable[None]] | None = None,
        user_id: str = "default",
        root_run_id: str | None = None,
        invocation_id: str | None = None,
        parent_invocation_id: str | None = None,
    ) -> tuple[str, str | None]:
        """Run a target agent that was pulled in via MCP delegation.

        Used by /internal/delegate. The agent runs under its own token (so it
        can itself delegate), with read/write access set by the caller. Output
        is forwarded to the event bus so 飞书 sees the delegated work; the text
        is returned to feed back to the calling agent as a tool result.
        """
        from memory.context_builder import ContextBuilder
        from memory.permissions import ActorContext
        from memory.scope import MemoryScope
        from mcp_tools.mcp_config import write_mcp_config, remove_mcp_config

        actor = _memory_actor(agent, user_id=user_id)
        scope = MemoryScope(
            domain="project" if project_id else "agent",
            scope_id=project_id or agent.agent_id,
            agent_id=agent.agent_id,
            tenant_id=actor.tenant_id,
        )
        _ctx = await ContextBuilder().build_memory_context(
            actor=actor,
            scope=scope,
            query=content,
            intent="recall",
            channel_id=channel_id,
            thread_id=thread_id,
        )
        memory_ctx = _ctx.rendered
        ws_dir = workspace_manager.get_dir(channel_id)
        bound_spec = getattr(agent, "_spec", None)

        # The delegated target also gets an MCP config (child token) + roster so
        # it can itself delegate recursively.
        child_mcp_path = write_mcp_config(delegate_token, mcp_capabilities)
        roster = await self._build_roster(project_id)

        ctx = InvokeContext(
            channel_id=channel_id,
            user_message=content,
            user_id=user_id,
            workspace_id=project_id,
            thread_id=thread_id,
            agent_role=_agent_role(agent),
            memory=memory_ctx,
            workspace_dir=ws_dir,
            workspace_access="write" if access == "write" else "read",
            spec=bound_spec,
            project_id=project_id,
            delegate_token=delegate_token,
            mcp_config_path=child_mcp_path,
            roster=roster,
            mcp_capabilities=list(mcp_capabilities or []),
            output_intent=dict(output_intent or {}),
            root_run_id=root_run_id,
            invocation_id=invocation_id,
            parent_invocation_id=parent_invocation_id,
        )

        collected: list[str] = []
        final_text: str | None = None
        error: str | None = None
        mutation_paths: list[str] = []
        pending_mutations: dict[str, str] = {}
        workspace_before = await asyncio.to_thread(snapshot_workspace, ws_dir) if project_id else None
        try:
            if on_progress:
                try:
                    await on_progress(
                        "stage",
                        {
                            "phase": "coding" if access == "write" else "researching",
                            "agent_id": agent.agent_id,
                            "agent_name": agent.name,
                        },
                    )
                except Exception:
                    pass
            async for etype, data in agent.invoke(ctx):
                changed = _mutation_path(etype, data, pending_mutations)
                if changed and changed not in mutation_paths:
                    mutation_paths.append(changed)
                if etype == "text_delta":
                    delta = data.get("delta", "")
                    collected.append(delta)
                    if on_progress:
                        try:
                            await on_progress(
                                "text",
                                {
                                    "phase": "coding" if access == "write" else "researching",
                                    "agent_id": agent.agent_id,
                                    "agent_name": agent.name,
                                    "delta": delta,
                                },
                            )
                        except Exception:
                            pass
                elif etype == "final_text":
                    value = data.get("text")
                    if isinstance(value, str) and value.strip():
                        final_text = _safe_final_message(value.strip())
                        if on_progress:
                            try:
                                await on_progress(
                                    "checkpoint",
                                    {
                                        "phase": "coding" if access == "write" else "researching",
                                        "agent_id": agent.agent_id,
                                        "agent_name": agent.name,
                                        "title": f"{agent.name} 已返回阶段结果",
                                        "detail": final_text[:500],
                                    },
                                )
                            except Exception:
                                pass
                    continue
                elif etype in ("tool_call", "tool_result") and on_progress:
                    try:
                        await on_progress(
                            etype,
                            {
                                **data,
                                "phase": "coding" if access == "write" else "researching",
                                "agent_id": agent.agent_id,
                                "agent_name": agent.name,
                            },
                        )
                    except Exception:
                        pass
                elif etype == "error":
                    error = str(data.get("message") or "agent failed")
                elif etype == "recoverable_error":
                    error = "RECOVERABLE:" + str(
                        data.get("message") or "agent execution stalled"
                    )
                await event_bus.publish(
                    channel_id, agent.agent_id, etype, _public_event_data(etype, data), agent_name=agent.name
                )
                if etype in ("done", "error", "recoverable_error"):
                    break
        except Exception as e:
            error = str(e)
            await event_bus.publish(
                channel_id, agent.agent_id, "error",
                {"message": sanitize_agent_text(str(e)).text}, agent_name=agent.name
            )
        finally:
            await event_bus.publish(
                channel_id, agent.agent_id, "status", {"state": "idle"}, agent_name=agent.name
            )
            # Clean up the per-run MCP config written for this delegated agent.
            if ctx.mcp_config_path:
                from mcp_tools.mcp_config import remove_mcp_config
                remove_mcp_config(ctx.mcp_config_path)

        detected_paths = await asyncio.to_thread(changed_workspace_paths, ws_dir, workspace_before)
        mutation_paths = _merge_changed_paths(mutation_paths, detected_paths)
        text = _safe_final_message(final_text or "".join(collected).strip())
        await _submit_invocation_memory(
            user_id=user_id,
            project_id=project_id,
            channel_id=channel_id,
            thread_id=thread_id,
            run_id=root_run_id,
            agent=agent,
            user_text=content,
            final_text=text,
            mutation_paths=mutation_paths,
            error=error,
        )
        return text, error

    async def _invoke_collect(
        self,
        channel_id: str,
        agent: BaseAgent,
        content: str,
        cancel_ev: asyncio.Event | None,
        *,
        thread_id: str | None = None,
        phase: str | None = None,
        run_id: str | None = None,
        on_progress: Callable[[str, dict], Awaitable[None]] | None = None,
        force_read_only: bool = False,
        project_id: str | None = None,
        spec: object | None = None,
        delegate_token: str | None = None,
        mcp_config_path: str | None = None,
        roster: str = "",
        mcp_capabilities: list[str] | None = None,
        output_intent: dict | None = None,
        user_id: str = "default",
        invocation_id: str | None = None,
        parent_invocation_id: str | None = None,
    ) -> tuple[str, str | None]:
        from memory.context_builder import ContextBuilder
        from memory.scope import MemoryScope

        history: list = []
        actor = _memory_actor(agent, user_id=user_id)
        scope = MemoryScope(
            domain="project" if project_id else "agent",
            scope_id=project_id or agent.agent_id,
            agent_id=agent.agent_id,
            tenant_id=actor.tenant_id,
        )
        _ctx = await ContextBuilder().build_memory_context(
            actor=actor,
            scope=scope,
            query=content,
            intent="recall",
            channel_id=channel_id,
            thread_id=thread_id,
        )
        memory_ctx = _ctx.rendered
        ws_dir = workspace_manager.get_dir(channel_id)

        # read-only tasks (or explicit read-only phases) never get write access.
        write_phase = phase in ("coding", "revising") and not force_read_only
        # Materialized project agents carry their spec; prefer it over an explicit arg.
        bound_spec = spec if spec is not None else getattr(agent, "_spec", None)
        ctx = InvokeContext(
            channel_id=channel_id,
            user_message=content,
            user_id=user_id,
            workspace_id=project_id,
            thread_id=thread_id,
            agent_role=_agent_role(agent),
            history=history,
            memory=memory_ctx,
            workspace_dir=ws_dir,
            workspace_access="write" if write_phase else "read",
            cancel_event=cancel_ev,
            spec=bound_spec,
            project_id=project_id,
            delegate_token=delegate_token,
            mcp_config_path=mcp_config_path,
            roster=roster,
            mcp_capabilities=list(mcp_capabilities or []),
            output_intent=dict(output_intent or {}),
            root_run_id=run_id,
            invocation_id=invocation_id or run_id,
            parent_invocation_id=parent_invocation_id,
        )

        status_data = {"state": "thinking"}
        if phase:
            status_data["phase"] = phase
        if run_id:
            status_data["run_id"] = run_id
        await event_bus.publish(channel_id, agent.agent_id, "status", status_data, agent_name=agent.name)
        if on_progress:
            try:
                await on_progress(
                    "stage",
                    {
                        "phase": phase or "entry",
                        "agent_id": agent.agent_id,
                        "agent_name": agent.name,
                        "run_id": run_id,
                    },
                )
            except Exception:
                pass
        agent.status = "thinking"

        collected: list[str] = []
        final_text: str | None = None
        error: str | None = None
        mutation_paths: list[str] = []
        pending_mutations: dict[str, str] = {}
        workspace_before = await asyncio.to_thread(snapshot_workspace, ws_dir) if project_id else None
        try:
            async for etype, data in agent.invoke(ctx):
                changed = _mutation_path(etype, data, pending_mutations)
                if changed and changed not in mutation_paths:
                    mutation_paths.append(changed)
                if cancel_ev and cancel_ev.is_set():
                    error = "cancelled"
                    await event_bus.publish(channel_id, agent.agent_id, "status", {"state": "idle"}, agent_name=agent.name)
                    break
                if etype == "text_delta":
                    collected.append(data.get("delta", ""))
                    if on_progress:
                        try:
                            await on_progress(
                                "text",
                                {"phase": phase, "agent_id": agent.agent_id, "agent_name": agent.name, "delta": data.get("delta", "")},
                            )
                        except Exception:
                            pass
                elif etype == "final_text":
                    value = data.get("text")
                    if isinstance(value, str) and value.strip():
                        final_text = _safe_final_message(value.strip())
                    # Internal collection hint only; do not put an unsupported
                    # event type on the public AgentEvent/WebSocket stream.
                    continue
                elif etype in ("tool_call", "tool_result") and on_progress:
                    try:
                        await on_progress(
                            etype,
                            {
                                **data,
                                "phase": phase,
                                "agent_id": agent.agent_id,
                                "agent_name": agent.name,
                            },
                        )
                    except Exception:
                        pass
                elif etype == "error":
                    error = str(data.get("message") or "agent failed")
                elif etype == "recoverable_error":
                    error = "RECOVERABLE:" + str(
                        data.get("message") or "agent execution stalled"
                    )
                await event_bus.publish(
                    channel_id, agent.agent_id, etype,
                    _public_event_data(etype, data), agent_name=agent.name,
                )
                if etype in ("done", "error", "recoverable_error"):
                    break
        except Exception as e:
            error = str(e)
            await event_bus.publish(
                channel_id, agent.agent_id, "error",
                {"message": sanitize_agent_text(str(e)).text}, agent_name=agent.name,
            )
        finally:
            agent.status = "idle"
            await event_bus.publish(channel_id, agent.agent_id, "status", {"state": "idle"}, agent_name=agent.name)

        detected_paths = await asyncio.to_thread(changed_workspace_paths, ws_dir, workspace_before)
        mutation_paths = _merge_changed_paths(mutation_paths, detected_paths)
        text = _safe_final_message(final_text or "".join(collected).strip())
        if text:
            await session_manager.append_message(
                channel_id,
                "agent",
                text,
                agent_id=agent.agent_id,
                thread_id=thread_id,
            )
        await _submit_invocation_memory(
            user_id=user_id,
            project_id=project_id,
            channel_id=channel_id,
            thread_id=thread_id,
            run_id=run_id,
            agent=agent,
            user_text=content,
            final_text=text,
            mutation_paths=mutation_paths,
            error=error,
        )
        return text, error

    async def run_entry(
        self,
        channel_id: str,
        thread_id: str,
        content: str,
        entry_agent: BaseAgent,
        trigger_message_id: str,
        run_id: str,
        on_status: Callable[[str, dict], Awaitable[None]] | None = None,
        on_progress: Callable[[str, dict], Awaitable[None]] | None = None,
        project_id: str | None = None,
        roster: str = "",
        output_intent: dict | None = None,
        user_id: str = "default",
        invocation_id: str | None = None,
        parent_invocation_id: str | None = None,
        record_user_message: bool = True,
    ) -> dict:
        """Run the entry agent, which drives collaboration via MCP delegation.

        This is the A-plan main path. The entry agent (default 橘长) receives the
        task + roster and decides itself whether to answer directly or pull other
        agents through the `delegate` tool. No fixed pipeline.
        """
        from core.delegate_registry import delegate_registry
        from mcp_tools.mcp_config import write_mcp_config, remove_mcp_config

        cancel_ev = asyncio.Event()
        self._thread_cancel[thread_id] = cancel_ev
        task = asyncio.current_task()
        if task:
            self._thread_tasks[thread_id] = task

        supported = set(
            getattr(
                getattr(entry_agent, "_spec", None),
                "mcp_capabilities",
                [
                    "delegate",
                    "report_progress",
                    "memory_search",
                    "wiki_read",
                    "wiki_list",
                    "wiki_write",
                    "wiki_append",
                ],
            )
        )
        requested = {"delegate", "report_progress", "memory_search"}
        requested.update((output_intent or {}).get("wiki_capabilities", []))
        mcp_capabilities = sorted(supported & requested)
        artifacts: list[dict] = []
        current_invocation_id = invocation_id or _uuid()

        token = await delegate_registry.issue(
            project_id=project_id,
            channel_id=channel_id,
            thread_id=thread_id,
            run_id=run_id,
            depth=0,
            caller_agent_id=entry_agent.agent_id,
            caller_agent_name=entry_agent.name,
            phase="entry",
            on_progress=on_progress,
            mcp_capabilities=mcp_capabilities,
            output_intent=dict(output_intent or {}),
            artifacts=artifacts,
            user_id=user_id,
            invocation_id=current_invocation_id,
            parent_invocation_id=parent_invocation_id,
        )
        mcp_path = write_mcp_config(token, mcp_capabilities)

        try:
            if record_user_message:
                await session_manager.append_message(
                    channel_id,
                    "user",
                    content,
                    thread_id=thread_id,
                    external_message_id=trigger_message_id,
                )
                from memory.memory_api import MemoryAPI
                history_actor = _memory_actor(entry_agent, user_id=user_id)
                try:
                    await MemoryAPI().history_append(
                        channel_id,
                        "user",
                        content,
                        actor=history_actor,
                        thread_id=thread_id,
                        message_id=f"sm_{uuid.uuid5(uuid.NAMESPACE_URL, trigger_message_id).hex[:16]}",
                    )
                except Exception:
                    pass

            outputs: dict[str, object] = {
                "mode": "entry",
                "entry_agent": entry_agent.agent_id,
                "output_intent": dict(output_intent or {}),
            }

            # Run the entry agent with its MCP delegation context + roster.
            text, error = await self._invoke_collect(
                channel_id,
                entry_agent,
                content,
                cancel_ev,
                thread_id=thread_id,
                phase="entry",
                run_id=run_id,
                on_progress=on_progress,
                project_id=project_id,
                delegate_token=token,
                mcp_config_path=mcp_path,
                roster=roster,
                mcp_capabilities=mcp_capabilities,
                output_intent=output_intent,
                user_id=user_id,
                invocation_id=current_invocation_id,
                parent_invocation_id=parent_invocation_id,
            )
            if error:
                if error.startswith("RECOVERABLE:"):
                    from core.exceptions import RecoverableRunError
                    raise RecoverableRunError(error.removeprefix("RECOVERABLE:"))
                raise RuntimeError(error)

            # delegate is asynchronous: when this invocation created work
            # orders, persist its stage result and end the process. The queue
            # will wake the same agent after every child has reached a terminal
            # state and will feed their results back in a fresh invocation.
            from core.run_store import run_store
            children = await run_store.children_for_parent(
                run_id, current_invocation_id
            )
            if children:
                outputs["suspended"] = True
                outputs["invocation_id"] = current_invocation_id
                outputs["summary"] = text
                outputs["pending_child_ids"] = [item.id for item in children]
                outputs["artifacts"] = list(artifacts)
                return outputs

            final_message = text or f"[{entry_agent.name}] 已完成。"
            outputs["summary"] = final_message
            outputs["final_message"] = final_message
            outputs["artifacts"] = list(artifacts)
            outputs["invocation_id"] = current_invocation_id
            from memory.memory_api import MemoryAPI, WriteProposal
            # Memory 2.2: conversation_completed trigger (entry-agent run end).
            if settings.memory_v22_enabled:
                try:
                    from memory.v22 import get_memory_extractor
                    from memory.v22.extractor import ExtractionContext
                    extractor = get_memory_extractor()
                    if extractor is not None:
                        await extractor.extract(ExtractionContext(
                            event_type="conversation_completed",
                            agent_id=entry_agent.agent_id,
                            agent_name=entry_agent.name,
                            user_text=content,
                            final_text=final_message,
                            mutation_paths=[],
                            succeeded=True,
                            run_id=run_id,
                            channel_id=channel_id,
                            thread_id=thread_id,
                            project_id=project_id,
                            user_id=user_id,
                        ))
                except Exception:
                    logger.exception("v22 memory extraction failed (non-fatal)")

            actor = _memory_actor(entry_agent, user_id=user_id)
            api = MemoryAPI()
            proposal = WriteProposal(
                text=final_message,
                domain="user",
                scope_hint=user_id,
                kind="project_fact",
                importance=0.5,
                confidence=0.5,
                source_type="agent_result",
                source_ids=[trigger_message_id] if trigger_message_id else [],
                actor_id=entry_agent.agent_id,
                event_type="conversation_completed",
                user_text=content,
                final_text=final_message,
                allowed_domains=["user"],
                channel_id=channel_id,
                thread_id=thread_id,
                run_id=run_id,
                agent_role=_agent_role(entry_agent),
                workspace_id=project_id,
            )
            try:
                await api.write_propose(proposal, actor=actor)
            except Exception:
                pass
            await event_bus.publish(
                channel_id,
                "system",
                "done",
                {"message": "entry complete", "run_id": run_id},
            )
            return outputs
        finally:
            remove_mcp_config(mcp_path)
            await delegate_registry.revoke(token)
            self._thread_tasks.pop(thread_id, None)
            self._thread_cancel.pop(thread_id, None)

    async def run_coordinator(
        self,
        channel_id: str,
        thread_id: str,
        content: str,
        trigger_message_id: str,
        run_id: str,
        on_status: Callable[[str, dict], Awaitable[None]] | None = None,
        intent: object | None = None,
        on_progress: Callable[[str, dict], Awaitable[None]] | None = None,
        project_id: str | None = None,
        user_id: str = "default",
    ) -> dict:
        """Execute Coordinator -> Research -> Coding -> Reviewer -> summary (fallback).

        Kept as a non-MCP fallback pipeline. The main path is run_entry, where
        the entry agent schedules peers via MCP delegation.
        """
        if thread_id in self._thread_tasks and not self._thread_tasks[thread_id].done():
            raise RuntimeError("a coordinator run is already active in this thread")

        cancel_ev = asyncio.Event()
        self._thread_cancel[thread_id] = cancel_ev
        task = asyncio.current_task()
        if task:
            self._thread_tasks[thread_id] = task

        async def set_status(status: str, outputs: dict) -> None:
            if on_status:
                await on_status(status, outputs)

        roles = resolve_role_agents()
        # Honor an explicit agent routing instruction from the (legacy) intent.
        if intent:
            preferred = getattr(intent, "preferred_agent_id", None)
            if preferred:
                routed = registry.get(preferred)
                if routed is not None:
                    roles["coding"] = routed
            read_only = bool(getattr(intent, "read_only", False) or not getattr(intent, "needs_coding", True))
        else:
            read_only = False

        outputs: dict[str, object] = {
            "roles": {role: agent.agent_id for role, agent in roles.items()}
        }
        if intent:
            outputs["intent"] = {
                "read_only": getattr(intent, "read_only", False),
                "needs_coding": getattr(intent, "needs_coding", True),
                "preferred_agent_id": getattr(intent, "preferred_agent_id", None),
                "summary": getattr(intent, "summary", ""),
                "method": getattr(intent, "method", ""),
            }
        await session_manager.append_message(
            channel_id,
            "user",
            content,
            thread_id=thread_id,
            external_message_id=trigger_message_id,
        )

        try:
            await set_status("planning", outputs)
            planning_prompt = (
                f"{ROLE_SYSTEM_GUIDANCE['coordinator']}\n\n"
                f"用户目标：\n{content}\n\n"
                "请先阅读当前项目上下文，把任务拆成 Research、Coding、Reviewer 可执行的步骤。"
                "输出一个简洁计划，说明目标、约束、子任务和验收条件。"
            )
            plan_text, error = await self._invoke_collect(
                channel_id,
                roles["coordinator"],
                planning_prompt,
                cancel_ev,
                thread_id=thread_id,
                phase="planning",
                run_id=run_id,
                on_progress=on_progress,
                project_id=project_id,
                force_read_only=read_only,
                user_id=user_id,
            )
            if error:
                raise RuntimeError(f"Coordinator 规划失败：{error}")
            outputs["plan"] = plan_text

            await set_status("researching", outputs)
            if read_only:
                research_prompt = (
                    f"{ROLE_SYSTEM_GUIDANCE['research']}\n\n"
                    f"用户目标：\n{content}\n\nCoordinator 计划：\n{plan_text}\n\n"
                    "这是一个只读检查任务。请阅读项目现状，给出事实、结构、问题和建议，"
                    "不要修改任何文件。"
                )
            else:
                research_prompt = (
                    f"{ROLE_SYSTEM_GUIDANCE['research']}\n\n"
                    f"用户目标：\n{content}\n\nCoordinator 计划：\n{plan_text}\n\n"
                    "请检查项目现状，给出实施所需的事实、可行方案、风险和建议。"
                )
            research_text, error = await self._invoke_collect(
                channel_id,
                roles["research"],
                research_prompt,
                cancel_ev,
                thread_id=thread_id,
                phase="researching",
                run_id=run_id,
                on_progress=on_progress,
                project_id=project_id,
                force_read_only=read_only,
                user_id=user_id,
            )
            if error:
                raise RuntimeError(f"Research Agent 执行失败：{error}")
            outputs["research"] = research_text

            coding_text = ""
            review_text = ""
            revisions: list[str] = []
            if not read_only:
                await set_status("coding", outputs)
                coding_prompt = (
                    f"{ROLE_SYSTEM_GUIDANCE['coding']}\n\n"
                    f"用户目标：\n{content}\n\nCoordinator 计划：\n{plan_text}\n\n"
                    f"Research 结论：\n{research_text}\n\n"
                    "请直接完成工作空间内的实现，并运行与改动风险相称的验证。"
                    "最后列出改动文件、验证结果和未解决事项。"
                )
                coding_text, error = await self._invoke_collect(
                    channel_id,
                    roles["coding"],
                    coding_prompt,
                    cancel_ev,
                    thread_id=thread_id,
                    phase="coding",
                    run_id=run_id,
                    on_progress=on_progress,
                    project_id=project_id,
                    user_id=user_id,
                )
                if error:
                    raise RuntimeError(f"Coding Agent 执行失败：{error}")
                outputs["coding"] = coding_text

                for revision_index in range(settings.coordinator_max_revisions + 1):
                    await set_status("reviewing", outputs)
                    review_prompt = (
                        f"{ROLE_SYSTEM_GUIDANCE['reviewer']}\n\n"
                        f"用户目标：\n{content}\n\nCoordinator 计划：\n{plan_text}\n\n"
                        f"Research 结论：\n{research_text}\n\nCoding 结果：\n{coding_text}\n\n"
                        "请检查工作空间中的实际改动和验证结果。通过则首行输出 VERDICT: PASS；"
                        "需要修正则首行输出 VERDICT: REVISE，并列出可执行问题。"
                    )
                    review_text, error = await self._invoke_collect(
                        channel_id,
                        roles["reviewer"],
                        review_prompt,
                        cancel_ev,
                        thread_id=thread_id,
                        phase="reviewing",
                        run_id=run_id,
                        on_progress=on_progress,
                        project_id=project_id,
                        user_id=user_id,
                    )
                    if error:
                        raise RuntimeError(f"Reviewer Agent 执行失败：{error}")
                    outputs["review"] = review_text
                    if self._review_passed(review_text):
                        break
                    if revision_index >= settings.coordinator_max_revisions:
                        outputs["review_warning"] = "达到最大修正次数，仍需人工检查。"
                        break

                    await set_status("revising", outputs)
                    revision_prompt = (
                        f"{ROLE_SYSTEM_GUIDANCE['coding']}\n\n"
                        f"用户目标：\n{content}\n\nReviewer 问题：\n{review_text}\n\n"
                        "请逐项修正 Reviewer 指出的问题，重新验证，并列出本轮修正。"
                    )
                    revision_text, error = await self._invoke_collect(
                        channel_id,
                        roles["coding"],
                        revision_prompt,
                        cancel_ev,
                        thread_id=thread_id,
                        phase="revising",
                        run_id=run_id,
                        on_progress=on_progress,
                        project_id=project_id,
                        user_id=user_id,
                    )
                    if error:
                        raise RuntimeError(f"Coding Agent 修正失败：{error}")
                    revisions.append(revision_text)
                    coding_text = f"{coding_text}\n\n修正第 {revision_index + 1} 轮：\n{revision_text}"
                    outputs["coding"] = coding_text
                    outputs["revisions"] = revisions

            await set_status("summarizing", outputs)
            if read_only:
                summary_prompt = (
                    f"{ROLE_SYSTEM_GUIDANCE['coordinator']}\n\n"
                    f"用户原始目标：\n{content}\n\n计划：\n{plan_text}\n\n"
                    f"Research 调查结果：\n{research_text}\n\n"
                    "这是只读检查任务，没有代码改动。请生成适合飞书群聊的最终回复，"
                    "简洁说明发现、现状和建议；不要输出内部思考过程。"
                )
            else:
                summary_prompt = (
                    f"{ROLE_SYSTEM_GUIDANCE['coordinator']}\n\n"
                    f"用户原始目标：\n{content}\n\n计划：\n{plan_text}\n\n"
                    f"Research：\n{research_text}\n\nCoding：\n{coding_text}\n\n"
                    f"Reviewer：\n{review_text}\n\n"
                    "请生成适合飞书群聊的最终回复。简洁说明完成了什么、验证结果、"
                    "仍需注意的事项；不要输出内部思考过程。"
                )
            summary, error = await self._invoke_collect(
                channel_id,
                roles["coordinator"],
                summary_prompt,
                cancel_ev,
                thread_id=thread_id,
                phase="summarizing",
                run_id=run_id,
                on_progress=on_progress,
                project_id=project_id,
                force_read_only=read_only,
                user_id=user_id,
            )
            if error:
                raise RuntimeError(f"Coordinator 汇总失败：{error}")
            outputs["summary"] = summary
            from memory.memory_api import MemoryAPI, WriteProposal
            # Memory 2.2: conversation_completed trigger (coordinator summary).
            if settings.memory_v22_enabled:
                try:
                    from memory.v22 import get_memory_extractor
                    from memory.v22.extractor import ExtractionContext
                    extractor = get_memory_extractor()
                    if extractor is not None:
                        await extractor.extract(ExtractionContext(
                            event_type="conversation_completed",
                            agent_id=roles["coordinator"].agent_id,
                            agent_name=roles["coordinator"].name,
                            user_text=content,
                            final_text=summary,
                            mutation_paths=[],
                            succeeded=True,
                            run_id=run_id,
                            channel_id=channel_id,
                            thread_id=thread_id,
                            project_id=project_id,
                            user_id=user_id,
                        ))
                except Exception:
                    logger.exception("v22 memory extraction failed (non-fatal)")

            actor = _memory_actor(roles["coordinator"], user_id=user_id)
            api = MemoryAPI()
            proposal = WriteProposal(
                text=summary,
                domain="user",
                scope_hint=user_id,
                kind="project_fact",
                importance=0.5,
                confidence=0.5,
                source_type="agent_result",
                source_ids=[trigger_message_id] if trigger_message_id else [],
                actor_id=roles["coordinator"].agent_id,
                event_type="conversation_completed",
                user_text=content,
                final_text=summary,
                allowed_domains=["user"],
                channel_id=channel_id,
                thread_id=thread_id,
                run_id=run_id,
                agent_role=_agent_role(roles["coordinator"]),
                workspace_id=project_id,
            )
            try:
                await api.write_propose(proposal, actor=actor)
            except Exception:
                pass
            await event_bus.publish(
                channel_id,
                "system",
                "done",
                {"message": "coordinator complete", "run_id": run_id},
            )
            return outputs
        finally:
            self._thread_tasks.pop(thread_id, None)
            self._thread_cancel.pop(thread_id, None)

    @staticmethod
    def _review_passed(text: str) -> bool:
        first_nonempty = next((line.strip() for line in text.splitlines() if line.strip()), "")
        return bool(re.match(r"^VERDICT\s*:\s*PASS\b", first_nonempty, re.IGNORECASE))


orchestrator = Orchestrator()
