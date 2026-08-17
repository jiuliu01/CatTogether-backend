"""Entry resolution: decide which agent is the entry point for a task.

  - Explicit naming ("让 暹罗 做…" / "@英短" / "用 Claude Code …") → that role.
  - No naming → the project's default agent, or the configured default entry
    agent (橘长 / coordinator), i.e. the 接单员.

Returns a single (entry_agent_id, task_text). Unlike the old router, it does
NOT enumerate multiple targets or decide execution order — that is now the
entry agent's job via MCP delegation.
"""
from __future__ import annotations

import re

from config import settings
from core.registry import registry


_TOOL_NAME_TO_AGENT_ID = {
    "claudecode": "claude-code",
    "claude": "claude-code",
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "")


async def _resolve_by_name(name: str, project_id: str | None = None) -> str | None:
    """Project agent specs first, then the global built-in registry."""
    normalized = _normalize_name(name)
    if not normalized:
        return None

    if project_id:
        from core.agent_store import agent_store
        try:
            specs = await agent_store.list(project_id)
        except Exception:
            specs = []
        for spec in specs:
            if getattr(spec, "internal", False):
                continue  # internal agents are not name-resolvable
            if _normalize_name(spec.name) == normalized or _normalize_name(spec.agent_id) == normalized:
                return spec.agent_id

    if registry.get(name.strip()):
        # Reject internal agents (e.g. memory) at the global layer too.
        from agents.memory_agent import is_internal
        if not is_internal(name.strip()):
            return name.strip()
    for agent in registry.all():
        if is_internal(agent.agent_id):
            continue
        if _normalize_name(agent.name) == normalized or _normalize_name(agent.agent_id) == normalized:
            return agent.agent_id
    return None


_LET_VERBS = r"做|干|处理|看|检查|实现|重构|修复|规划|分析|设计|写|改|跑|测|部署|调试|排查"
# An agent name may be ASCII (Coder) or Chinese (暹罗/英短/橘长).
_NAME = r"[A-Za-z一-鿿][\w一-鿿\- ]*?"
# @name must stop at a space/punctuation boundary so it doesn't swallow the
# trailing task text. [\w一-鿿\-] alone (no space) keeps the name contiguous.
_NAME_AT = r"[A-Za-z一-鿿][\w一-鿿\-]*"


async def _explicit_target(content: str, project_id: str | None = None) -> str | None:
    """Return the single agent_id explicitly named, or None."""
    let_match = re.search(
        rf"让\s+({_NAME})(?:来\s*)?(?:{_LET_VERBS})",
        content,
        re.UNICODE,
    )
    if let_match:
        # "让 A 和 B 做…" — pick the first named agent as entry.
        span = let_match.group(1)
        for part in re.split(r"\s*(?:和|、|,|，)\s*", span):
            aid = await _resolve_by_name(part, project_id)
            if aid:
                return aid

    # @name — accept ASCII or Chinese name chars (contiguous, no spaces).
    for m in re.finditer(rf"@({_NAME_AT})", content):
        aid = await _resolve_by_name(m.group(1), project_id)
        if aid:
            return aid

    tool_match = re.search(r"(?:用|使用)\s*(Claude\s*Code|claude)", content, re.UNICODE)
    if tool_match:
        key = _normalize_name(tool_match.group(1))
        aid = _TOOL_NAME_TO_AGENT_ID.get(key)
        if aid and registry.get(aid):
            return aid

    return None


def strip_routing_prefix(content: str) -> str:
    text = content
    text = re.sub(rf"^让\s+{_NAME}(?:来\s*)?(?={_LET_VERBS})", "", text, count=1, flags=re.UNICODE)
    text = re.sub(rf"^@{_NAME_AT}\s+", "", text, count=1, flags=re.UNICODE)
    text = re.sub(r"^(?:用|使用)\s*(?:Claude\s*Code|claude)\s+", "", text, count=1, flags=re.UNICODE)
    return text.strip() or content.strip()


def _is_explicit(content: str) -> bool:
    """True if the message names an agent explicitly (prefix should be stripped)."""
    if re.match(rf"^让\s+{_NAME}(?:来\s*)?(?:{_LET_VERBS})", content, re.UNICODE):
        return True
    if re.match(rf"^@{_NAME_AT}\s+", content, re.UNICODE):
        return True
    if re.match(r"^(?:用|使用)\s*(?:Claude\s*Code|claude)\s+", content, re.UNICODE):
        return True
    return False


async def resolve(
    content: str,
    project_id: str | None = None,
    mentioned: list[str] | None = None,
) -> tuple[str, str]:
    """Return (entry_agent_id, task_text).

    mentioned (from the Feishu parser, agent_ids) wins first; then explicit
    naming in the text; otherwise the project default / 接单员.
    """
    target: str | None = None
    if mentioned:
        from agents.memory_agent import is_internal
        for aid in mentioned:
            if registry.get(aid) and not is_internal(aid):
                target = aid
                break

    if not target:
        target = await _explicit_target(content, project_id)

    if not target:
        # No explicit name → 接单员: project default, else configured default.
        if project_id:
            from core.project_store import project_store
            project = await project_store.get(project_id)
            if project and project.default_agent_id:
                target = project.default_agent_id
        if not target:
            target = settings.default_entry_agent or "coordinator"

    task_text = strip_routing_prefix(content) if target and _is_explicit(content) else content
    return target, task_text
