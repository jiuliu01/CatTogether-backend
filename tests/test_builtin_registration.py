from agents.base import BaseAgent
from agents.roles import resolve_role_agents
from bootstrap import register_builtins
from config import settings
from core.registry import AgentRegistry


class StubAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, agent_id, "custom")

    async def invoke(self, ctx):
        yield ("done", {})


def test_register_builtins_only_keeps_claude_code(monkeypatch):
    import bootstrap

    isolated_registry = AgentRegistry()
    for agent_id in ("codex", "openai-default", "anthropic-default"):
        isolated_registry.register(StubAgent(agent_id))

    monkeypatch.setattr(bootstrap, "registry", isolated_registry)
    monkeypatch.setattr(settings, "openai_api_key", "configured")
    monkeypatch.setattr(settings, "anthropic_api_key", "configured")

    register_builtins()

    assert [agent.agent_id for agent in isolated_registry.all()] == ["claude-code"]


def test_all_default_roles_use_claude_code(monkeypatch):
    import agents.roles as roles
    from agents.cli.claude_code import ClaudeCodeAgent

    isolated_registry = AgentRegistry()
    isolated_registry.register(ClaudeCodeAgent())
    monkeypatch.setattr(roles, "registry", isolated_registry)
    monkeypatch.setattr(settings, "coordinator_agent_id", None)
    monkeypatch.setattr(settings, "research_agent_id", None)
    monkeypatch.setattr(settings, "coding_agent_id", None)
    monkeypatch.setattr(settings, "reviewer_agent_id", None)

    resolved = resolve_role_agents()

    assert set(resolved) == {"coordinator", "research", "coding", "reviewer"}
    assert {agent.agent_id for agent in resolved.values()} == {"claude-code"}
