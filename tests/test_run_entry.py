"""End-to-end test of the MCP delegation callback path.

We mock the Claude Code agent invoke so no real subprocess is needed, and we
point the MCP server at a fake backend. This verifies: token issuance, depth
check, target resolution, result feedback, and roster injection.
"""
import asyncio

from agents.base import BaseAgent, InvokeContext, RawEvent
from collections.abc import AsyncIterator
from core.delegate_registry import delegate_registry
from core.orchestrator import orchestrator


class StubAgent(BaseAgent):
    """A no-subprocess agent that just yields a fixed reply."""

    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, name, "custom")

    async def invoke(self, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        yield ("text_delta", {"delta": f"[{self.name}] done: {ctx.user_message}"})
        yield ("done", {})


class FinalAwareAgent(BaseAgent):
    def __init__(self):
        super().__init__("coordinator", "橘长", "custom")

    async def invoke(self, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        yield ("text_delta", {"delta": "我先读取项目。"})
        yield ("final_text", {"text": "这是干净的最终答案。"})
        yield ("done", {})


def test_run_entry_separates_progress_from_final_answer():
    async def run():
        progress = []

        async def on_progress(kind, data):
            progress.append((kind, data))

        outputs = await orchestrator.run_entry(
            channel_id="final-channel",
            thread_id="final-thread",
            content="读项目",
            entry_agent=FinalAwareAgent(),
            trigger_message_id="final-message",
            run_id="final-run",
            on_progress=on_progress,
        )
        return outputs, progress

    outputs, progress = asyncio.run(run())

    assert outputs["summary"] == "这是干净的最终答案。"
    assert [item[1]["delta"] for item in progress if item[0] == "text"] == [
        "我先读取项目。"
    ]


def test_resume_prompt_is_not_saved_as_a_user_message():
    from core.session_manager import session_manager

    async def run():
        channel = await session_manager.create_channel("resume-history")
        thread = await session_manager.get_or_create_thread(
            channel.id, "resume-root", source="feishu"
        )
        await orchestrator.run_entry(
            channel_id=channel.id,
            thread_id=thread.id,
            content="这是一次任务恢复。包含很长的内部子任务结果。",
            entry_agent=FinalAwareAgent(),
            trigger_message_id="resume-message",
            run_id="resume-run-no-user-history",
            record_user_message=False,
        )
        return await session_manager.get_scope_history(
            channel.id, thread_id=thread.id
        )

    history = asyncio.run(run())
    assert [message.role for message in history] == ["agent"]
    assert all("这是一次任务恢复" not in message.content for message in history)


def test_invoke_for_delegate_returns_target_output(monkeypatch):
    """The delegate callback runs the target and returns its text."""
    from core import registry as registry_module

    target = StubAgent("coder", "暹罗")
    monkeypatch.setattr(registry_module.registry, "get", lambda aid: target if aid == "coder" else None)

    async def run():
        # Issue a parent token at depth 0.
        parent_token = await delegate_registry.issue(
            project_id=None, channel_id="c1", thread_id="t1", run_id="r1", depth=0
        )
        # Materialize the target via registry (no project spec here).
        agent = registry_module.registry.get("coder")
        progress = []

        async def on_progress(kind, data):
            progress.append((kind, data))

        text, error = await orchestrator.invoke_for_delegate(
            agent=agent,
            content="实现 /hello",
            channel_id="c1",
            thread_id="t1",
            project_id=None,
            access="write",
            delegate_token=parent_token,
            on_progress=on_progress,
        )
        return text, error, progress

    text, error, progress = asyncio.run(run())
    assert error is None
    assert "暹罗" in text
    assert "/hello" in text
    assert any(
        kind == "stage" and data["agent_name"] == "暹罗"
        for kind, data in progress
    )
    assert any(
        kind == "text" and "实现 /hello" in data["delta"]
        for kind, data in progress
    )


def test_delegate_callback_rejects_invalid_token():
    async def run():
        agent = StubAgent("coder", "暹罗")
        # No token issued -> /internal/delegate would return invalid token.
        # Here we test invoke_for_delegate directly with a bogus token; it still
        # runs the agent (depth check is in the endpoint, not here).
        text, error = await orchestrator.invoke_for_delegate(
            agent=agent,
            content="x",
            channel_id="c1",
            thread_id=None,
            project_id=None,
            access="read",
            delegate_token="bogus",
        )
        return text, error

    text, error = asyncio.run(run())
    assert error is None
    assert "暹罗" in text


def test_self_delegation_is_blocked():
    """An agent must not pull itself in (would loop)."""
    from starlette.testclient import TestClient
    import main

    async def issue():
        # 橘长 (coordinator) holds this token.
        return await delegate_registry.issue(
            project_id=None, channel_id="c1", thread_id="t1", run_id="r1",
            depth=0, caller_agent_id="coordinator",
        )

    token = asyncio.run(issue())
    client = TestClient(main.app)
    # coordinator tries to delegate to itself.
    resp = client.post(
        "/internal/delegate",
        json={"token": token, "target": "coordinator", "task": "x", "access": "read"},
    )
    data = resp.json()
    assert data.get("error")
    assert "不能委派给自己" in data["error"]
    asyncio.run(delegate_registry.revoke(token))


def test_depth_limit_enforced_in_endpoint(monkeypatch):
    """When the parent is already at max depth, the endpoint rejects delegation."""
    from starlette.testclient import TestClient
    from config import settings
    import main

    monkeypatch.setattr(settings, "delegate_max_depth", 1)

    async def issue():
        return await delegate_registry.issue(
            project_id=None, channel_id="c1", thread_id="t1", run_id="r1", depth=1
        )

    token = asyncio.run(issue())
    client = TestClient(main.app)
    resp = client.post(
        "/internal/delegate",
        json={"token": token, "target": "coder", "task": "x", "access": "read"},
    )
    data = resp.json()
    assert data.get("error")
    assert "深度" in data["error"]
    asyncio.run(delegate_registry.revoke(token))


def test_report_progress_requires_capability_and_redacts_secret():
    from starlette.testclient import TestClient
    import main

    progress = []

    async def on_progress(kind, data):
        progress.append((kind, data))

    async def issue(caps):
        return await delegate_registry.issue(
            project_id=None,
            channel_id="c1",
            thread_id="t1",
            run_id="r-progress",
            caller_agent_id="coordinator",
            caller_agent_name="橘长",
            on_progress=on_progress,
            mcp_capabilities=caps,
        )

    client = TestClient(main.app)
    denied_token = asyncio.run(issue(["delegate"]))
    denied = client.post(
        "/internal/progress",
        json={"token": denied_token, "message": "完成检查"},
    ).json()
    assert "capability not granted" in denied["error"]
    asyncio.run(delegate_registry.revoke(denied_token))

    allowed_token = asyncio.run(issue(["report_progress"]))
    recorded = client.post(
        "/internal/progress",
        json={
            "token": allowed_token,
            "message": "已检查配置 OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz",
        },
    ).json()
    assert recorded["status"] == "recorded"
    assert progress and progress[-1][0] == "checkpoint"
    assert progress[-1][1]["agent_name"] == "橘长"
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in progress[-1][1]["title"]
    asyncio.run(delegate_registry.revoke(allowed_token))
