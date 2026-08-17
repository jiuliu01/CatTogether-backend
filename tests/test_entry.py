import asyncio

from core import entry as entry_module
from core.entry import resolve, strip_routing_prefix, _explicit_target, _is_explicit
from core.registry import registry
from agents.base import BaseAgent, InvokeContext, RawEvent
from collections.abc import AsyncIterator


class StubAgent(BaseAgent):
    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, name, "custom")

    async def invoke(self, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        yield ("done", {})


def _register(*agents: BaseAgent) -> None:
    for agent in agents:
        registry.register(agent)


def test_no_name_resolves_to_default_entry(monkeypatch):
    async def run():
        _register(StubAgent("coordinator", "橘长"))
        aid, task = await resolve("检查目录结构", project_id=None)
        return aid, task

    aid, task = asyncio.run(run())
    assert aid == "coordinator"
    assert task == "检查目录结构"  # no prefix stripped


def test_explicit_let_resolves_to_named_agent():
    async def run():
        _register(StubAgent("coder", "暹罗"))
        return await resolve("让 暹罗 重构 auth", project_id=None)

    aid, task = asyncio.run(run())
    assert aid == "coder"
    assert "让" not in task and "暹罗" not in task


def test_explicit_at_mention_resolves():
    async def run():
        _register(StubAgent("reviewer", "英短"))
        return await resolve("@英短 看一下改动", project_id=None)

    aid, task = asyncio.run(run())
    assert aid == "reviewer"
    assert "看一下改动" in task


def test_explicit_tool_routing_claude_code():
    async def run():
        _register(StubAgent("claude-code", "Claude Code"))
        return await resolve("用 Claude Code 加接口", project_id=None)

    aid, task = asyncio.run(run())
    assert aid == "claude-code"
    assert "加接口" in task


def test_unknown_name_falls_back_to_default(monkeypatch):
    async def run():
        _register(StubAgent("coordinator", "橘长"))
        # "让 蓝猫 做…" — 蓝猫 not registered -> default entry
        return await resolve("让 蓝猫 做事", project_id=None)

    aid, task = asyncio.run(run())
    assert aid == "coordinator"


def test_strip_routing_prefix():
    assert strip_routing_prefix("让 暹罗 重构 auth") == "重构 auth"
    assert strip_routing_prefix("@英短 看一下") == "看一下"
    assert strip_routing_prefix("帮我规划") == "帮我规划"


def test_is_explicit():
    assert _is_explicit("让 暹罗 重构 auth") is True
    assert _is_explicit("@英短 看") is True
    assert _is_explicit("用 Claude Code 加接口") is True
    assert _is_explicit("检查目录") is False
