"""Resolve logical collaboration roles to registered agents."""
from __future__ import annotations

from agents.base import BaseAgent
from config import settings
from core.registry import registry


class RoleResolutionError(RuntimeError):
    pass


def _configured(agent_id: str | None) -> BaseAgent | None:
    return registry.get(agent_id) if agent_id else None


def _first(*agent_ids: str | None) -> BaseAgent | None:
    for agent_id in agent_ids:
        agent = _configured(agent_id)
        if agent:
            return agent
    return None


def resolve_role_agents() -> dict[str, BaseAgent]:
    all_agents = registry.all()
    if not all_agents:
        raise RoleResolutionError("no agents are registered")

    coordinator = (
        _configured(settings.coordinator_agent_id)
        or _first("claude-code")
        or all_agents[0]
    )
    research = (
        _configured(settings.research_agent_id)
        or _first("claude-code")
        or coordinator
    )
    coding = (
        _configured(settings.coding_agent_id)
        or _first("claude-code")
        or coordinator
    )
    reviewer = (
        _configured(settings.reviewer_agent_id)
        or _first("claude-code")
        or coordinator
    )
    return {
        "coordinator": coordinator,
        "research": research,
        "coding": coding,
        "reviewer": reviewer,
    }


ROLE_SYSTEM_GUIDANCE = {
    "coordinator": (
        "你是 Coordinator。只根据用户目标和项目实际情况拆解、组织和汇总任务。"
        "输出要清晰、可执行，不虚构已完成的工作。"
    ),
    "research": (
        "你是 Research Agent。本阶段只读项目和资料，不修改文件。"
        "输出事实、方案、风险和对 Coding Agent 有用的结论。"
    ),
    "coding": (
        "你是 Coding Agent。根据计划在当前工作空间内完成必要修改并验证。"
        "不要操作工作空间之外的文件，不执行不可逆的高风险操作。"
    ),
    "reviewer": (
        "你是 Reviewer Agent。独立检查需求覆盖、实现质量、安全和验证结果。"
        "首行必须严格输出 VERDICT: PASS 或 VERDICT: REVISE。"
    ),
}
