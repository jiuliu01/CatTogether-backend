from collections.abc import AsyncIterator
import asyncio
from types import SimpleNamespace

from agents.base import BaseAgent, InvokeContext, RawEvent
from config import settings
from core.orchestrator import orchestrator
from core.registry import registry
from core.session_manager import session_manager
from memory.context_builder import ContextBuilder, MemoryContext
from memory.memory_api import MemoryAPI


class ScriptedAgent(BaseAgent):
    def __init__(self, agent_id: str, role: str):
        super().__init__(agent_id, agent_id, "custom")
        self.role = role
        self.calls = 0

    async def invoke(self, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        self.calls += 1
        if self.role == "coordinator":
            text = "最终汇总：功能已完成并通过复核。" if "最终回复" in ctx.user_message else "计划：调研、实现、复核。"
        elif self.role == "research":
            text = "调研结论：现有模块可以复用。"
        elif self.role == "coding":
            text = "已完成 Reviewer 要求的修正。" if "Reviewer 问题" in ctx.user_message else "实现完成并已验证。"
        else:
            text = "VERDICT: REVISE\n请补充验证。" if self.calls == 1 else "VERDICT: PASS\n复核通过。"
        yield ("text_delta", {"delta": text})
        yield ("done", {})


def test_coordinator_runs_all_roles_and_one_revision(monkeypatch):
    async def scenario():
        agents = {
            "coordinator": ScriptedAgent("test-coordinator", "coordinator"),
            "research": ScriptedAgent("test-research", "research"),
            "coding": ScriptedAgent("test-coding", "coding"),
            "reviewer": ScriptedAgent("test-reviewer", "reviewer"),
        }
        for agent in agents.values():
            registry.register(agent)

        monkeypatch.setattr(settings, "coordinator_agent_id", "test-coordinator")
        monkeypatch.setattr(settings, "research_agent_id", "test-research")
        monkeypatch.setattr(settings, "coding_agent_id", "test-coding")
        monkeypatch.setattr(settings, "reviewer_agent_id", "test-reviewer")
        monkeypatch.setattr(settings, "coordinator_max_revisions", 2)

        async def no_memory(*args, **kwargs):
            return MemoryContext()

        async def no_write(*args, **kwargs):
            return True

        # Patch the v2.1 context builder and write API to no-ops so the
        # coordinator test runs without a live memory DB.
        monkeypatch.setattr(ContextBuilder, "build_memory_context", no_memory)
        monkeypatch.setattr(MemoryAPI, "write_propose", no_write)

        channel = await session_manager.create_channel("test")
        thread = await session_manager.get_or_create_thread(
            channel.id, "om_root", source="feishu"
        )
        statuses = []

        async def on_status(status, outputs):
            statuses.append(status)

        outputs = await orchestrator.run_coordinator(
            channel.id,
            thread.id,
            "实现飞书接入",
            "om_trigger",
            "run_test",
            on_status,
        )

        assert outputs["summary"].startswith("最终汇总")
        assert outputs["review"].startswith("VERDICT: PASS")
        assert agents["coding"].calls == 2
        assert agents["reviewer"].calls == 2
        assert "revising" in statuses

        history = await session_manager.get_thread_history(thread.id)
        assert history[0].external_message_id == "om_trigger"

    asyncio.run(scenario())


def test_coordinator_read_only_intent_skips_coding(monkeypatch):
    async def scenario():
        agents = {
            "coordinator": ScriptedAgent("ro-coordinator", "coordinator"),
            "research": ScriptedAgent("ro-research", "research"),
            "coding": ScriptedAgent("ro-coding", "coding"),
            "reviewer": ScriptedAgent("ro-reviewer", "reviewer"),
        }
        for agent in agents.values():
            registry.register(agent)

        monkeypatch.setattr(settings, "coordinator_agent_id", "ro-coordinator")
        monkeypatch.setattr(settings, "research_agent_id", "ro-research")
        monkeypatch.setattr(settings, "coding_agent_id", "ro-coding")
        monkeypatch.setattr(settings, "reviewer_agent_id", "ro-reviewer")
        monkeypatch.setattr(settings, "coordinator_max_revisions", 2)

        async def no_memory(*args, **kwargs):
            return MemoryContext()

        async def no_write(*args, **kwargs):
            return True

        monkeypatch.setattr(ContextBuilder, "build_memory_context", no_memory)
        monkeypatch.setattr(MemoryAPI, "write_propose", no_write)

        channel = await session_manager.create_channel("ro-test")
        thread = await session_manager.get_or_create_thread(
            channel.id, "om_root_ro", source="feishu"
        )
        statuses = []

        async def on_status(status, outputs):
            statuses.append(status)

        intent = SimpleNamespace(
            read_only=True,
            needs_coding=False,
            preferred_agent_id=None,
            summary="检查项目",
            method="merged",
        )
        outputs = await orchestrator.run_coordinator(
            channel.id,
            thread.id,
            "检查项目结构",
            "om_trigger_ro",
            "run_ro",
            on_status,
            intent=intent,
        )

        assert outputs["summary"].startswith("最终汇总")
        assert "coding" not in outputs
        assert "review" not in outputs
        assert agents["coding"].calls == 0
        assert agents["reviewer"].calls == 0
        assert agents["research"].calls == 1
        assert "coding" not in statuses
        assert "reviewing" not in statuses

    asyncio.run(scenario())
