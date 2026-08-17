"""Agent registry: in-memory dict of agent_id -> BaseAgent.

Claude Code is registered at startup. Custom agents can be registered at
runtime via REST.
"""
from __future__ import annotations

import asyncio
from typing import Iterator

from agents.base import BaseAgent
from models.schemas import AgentInfo


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.agent_id] = agent

    def unregister(self, agent_id: str) -> bool:
        return self._agents.pop(agent_id, None) is not None

    def get(self, agent_id: str) -> BaseAgent | None:
        return self._agents.get(agent_id)

    def all(self) -> list[BaseAgent]:
        return list(self._agents.values())

    def infos(self) -> list[AgentInfo]:
        return [a.info() for a in self._agents.values()]

    def __iter__(self) -> Iterator[BaseAgent]:
        return iter(self._agents.values())


registry = AgentRegistry()


def materialize(spec) -> BaseAgent:
    """Build a ClaudeCodeAgent bound to an AgentSpec.

    The agent_id/name come from the spec so署名 and routing use the project
    agent's identity. The spec is attached as ``_spec`` so the executor can
    drive the CLI from it without callers4 passing it through every call site.
    """
    from agents.cli.claude_code import ClaudeCodeAgent

    agent = ClaudeCodeAgent(
        agent_id=spec.agent_id,
        name=spec.name,
        description=f"{spec.role} agent (project {spec.project_id})",
    )
    agent._spec = spec  # type: ignore[attr-defined]
    return agent
