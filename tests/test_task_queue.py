import asyncio
from types import SimpleNamespace

from agents.base import BaseAgent, InvokeContext, RawEvent
from collections.abc import AsyncIterator
from config import settings
from core import task_queue as task_queue_module
from core.session_manager import session_manager
from core.task_queue import FeishuTaskQueue
from models.schemas import AgentRun, FeishuBinding, FeishuInboundMessage


class StubEntryAgent(BaseAgent):
    def __init__(self):
        super().__init__("coordinator", "橘长", "custom")

    async def invoke(self, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        yield ("text_delta", {"delta": "我直接回答。"})
        yield ("done", {})


class FakeSender:
    def __init__(self):
        self.replies = []

    async def reply(self, message_id, text, **kwargs):
        self.replies.append((message_id, text, kwargs))
        return []


class FakeMappings:
    def __init__(self, binding):
        self.binding = binding

    async def get_binding(self, tenant_key, chat_id):
        return self.binding

    async def ensure_binding(self, inbound):
        return self.binding


class FakeRunStore:
    def __init__(self):
        self.run = None

    async def create(self, channel_id, thread_id, trigger_message_id, trigger_user_id=None):
        self.run = AgentRun(
            id="run_queue_test",
            channel_id=channel_id,
            thread_id=thread_id,
            trigger_message_id=trigger_message_id,
            trigger_user_id=trigger_user_id,
        )
        return self.run

    async def update(self, run_id, **values):
        for key, value in values.items():
            if value is not None:
                setattr(self.run, key, value)
        return self.run


class FakeOrchestrator:
    """Records calls; run_entry returns a signed summary like the real one."""

    def __init__(self, final_message="[橘长] 已完成任务。", artifacts=None):
        self.entry_called = False
        self.roster_called = False
        self.final_message = final_message
        self.artifacts = list(artifacts or [])

    async def _build_roster(self, project_id):
        self.roster_called = True
        return "roster"

    async def run_entry(self, **kwargs):
        self.entry_called = True
        on_progress = kwargs.get("on_progress")
        if on_progress:
            await on_progress("stage", {"phase": "entry", "agent_name": "橘长"})
        await kwargs["on_status"]("running", {})
        return {
            "mode": "entry",
            "entry_agent": "coordinator",
            "summary": self.final_message,
            "final_message": self.final_message,
            "artifacts": self.artifacts,
        }


class FakeEntry:
    def __init__(self, entry_id, task_text):
        self._entry_id = entry_id
        self._task_text = task_text

    async def resolve(self, content, project_id=None, mentioned=None):
        return self._entry_id, self._task_text


def _make_binding(channel_id):
    return FeishuBinding(
        id="binding-test",
        tenant_key="tenant",
        chat_id="chat",
        channel_id=channel_id,
        workspace_dir=str(settings.project_root),
    )


def test_queue_runs_entry_and_replies(monkeypatch):
    async def scenario():
        channel = await session_manager.create_channel("queue-test")
        binding = _make_binding(channel.id)
        runs = FakeRunStore()
        sender = FakeSender()
        fake_orch = FakeOrchestrator()
        monkeypatch.setattr(task_queue_module, "feishu_mapping_store", FakeMappings(binding))
        monkeypatch.setattr(task_queue_module, "run_store", runs)
        monkeypatch.setattr(task_queue_module, "orchestrator", fake_orch)
        monkeypatch.setattr(task_queue_module, "entry", FakeEntry("coordinator", "检查目录"))
        # _resolve_entry_agent calls registry.get; return a stub entry agent.
        from core import registry as registry_module
        monkeypatch.setattr(registry_module.registry, "get", lambda aid: StubEntryAgent())

        queue = FeishuTaskQueue(sender=sender)
        await queue.start()
        try:
            await queue.enqueue(
                FeishuInboundMessage(
                    message_id="om_queue_test",
                    tenant_key="tenant",
                    chat_id="chat",
                    sender_open_id="ou_user",
                    content="检查目录结构",
                    mentioned_bot=True,
                )
            )
            await queue._queue.join()
            await asyncio.sleep(0)
        finally:
            await queue.stop()

        assert runs.run.status == "completed", runs.run.error
        assert fake_orch.entry_called and fake_orch.roster_called
        assert runs.run.outputs["reply_agent_id"] == "coordinator"
        assert runs.run.outputs["reply_agent_name"] == "橘长"
        assert runs.run.outputs["process"][0]["agent_name"] == "橘长"
        assert any("已接单" in r[1] for r in sender.replies)
        assert any("已完成任务" in r[1] for r in sender.replies)

    asyncio.run(scenario())


def test_queue_reports_failure_when_entry_agent_missing(monkeypatch):
    async def scenario():
        channel = await session_manager.create_channel("queue-fail")
        binding = _make_binding(channel.id)
        runs = FakeRunStore()
        sender = FakeSender()
        monkeypatch.setattr(task_queue_module, "feishu_mapping_store", FakeMappings(binding))
        monkeypatch.setattr(task_queue_module, "run_store", runs)
        monkeypatch.setattr(task_queue_module, "orchestrator", FakeOrchestrator())
        # entry resolves to an id that _resolve_entry_agent can't materialize
        monkeypatch.setattr(task_queue_module, "entry", FakeEntry("nonexistent", "x"))
        # registry.get returns None for it
        from core import registry as registry_module
        monkeypatch.setattr(registry_module.registry, "get", lambda aid: None)

        queue = FeishuTaskQueue(sender=sender)
        await queue.start()
        try:
            await queue.enqueue(
                FeishuInboundMessage(
                    message_id="om_fail_test",
                    tenant_key="tenant",
                    chat_id="chatf",
                    sender_open_id="ou_user",
                    content="x",
                    mentioned_bot=True,
                )
            )
            await queue._queue.join()
            await asyncio.sleep(0)
        finally:
            await queue.stop()

        assert runs.run.status == "failed"
        assert any("任务执行失败" in r[1] for r in sender.replies)

    asyncio.run(scenario())


def test_long_chat_answer_is_not_turned_into_wiki(monkeypatch):
    async def scenario():
        channel = await session_manager.create_channel("queue-long-chat")
        binding = _make_binding(channel.id)
        runs = FakeRunStore()
        sender = FakeSender()
        long_answer = "这是普通聊天答案。" * 400
        fake_orch = FakeOrchestrator(final_message=long_answer)
        monkeypatch.setattr(task_queue_module, "feishu_mapping_store", FakeMappings(binding))
        monkeypatch.setattr(task_queue_module, "run_store", runs)
        monkeypatch.setattr(task_queue_module, "orchestrator", fake_orch)
        monkeypatch.setattr(task_queue_module, "entry", FakeEntry("coordinator", "详细分析这个问题"))
        from core import registry as registry_module
        monkeypatch.setattr(registry_module.registry, "get", lambda aid: StubEntryAgent())

        queue = FeishuTaskQueue(sender=sender)
        await queue.start()
        try:
            await queue.enqueue(
                FeishuInboundMessage(
                    message_id="om_long_chat",
                    tenant_key="tenant",
                    chat_id="chat-long",
                    sender_open_id="ou_user",
                    content="详细分析这个问题",
                    mentioned_bot=True,
                )
            )
            await queue._queue.join()
            await asyncio.sleep(0)
        finally:
            await queue.stop()

        assert len(runs.run.outputs["final_message"]) > 2000
        assert runs.run.outputs["artifacts"] == []
        assert runs.run.outputs["archive_status"] == "not_requested"
        assert runs.run.intent["mode"] == "chat"

    asyncio.run(scenario())
