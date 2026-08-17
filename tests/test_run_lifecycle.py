import asyncio
import json
import os
from collections.abc import AsyncIterator
from types import SimpleNamespace

from agents.base import BaseAgent, InvokeContext, RawEvent
from config import settings
from core.run_store import RunStore
from core.run_event_store import RunEventStore
from core.run_supervisor import RunSupervisor
from models.schemas import DelegationRecord


class DelegateTarget(BaseAgent):
    def __init__(self):
        super().__init__("coder", "开发角色", "custom")

    async def invoke(self, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        raise AssertionError("delegate endpoint must not run the child synchronously")
        yield ("done", {})


def test_unfinished_run_is_kept_for_restart_recovery(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    async def scenario():
        store = RunStore()
        run = await store.create("channel", "thread", "message")
        await store.update(
            run.id,
            status="coding",
            original_task="完成微信小程序功能",
            workspace_dir=str(tmp_path / "life-note-miniapp"),
            root_invocation_id="root-1",
        )
        await store.add_delegation(
            run.id,
            DelegationRecord(
                id="child-1",
                parent_invocation_id="root-1",
                target_agent_id="coder",
                task="实现页面",
                access="write",
            ),
        )
        return run.id

    run_id = asyncio.run(scenario())
    restored = RunStore()
    run = asyncio.run(restored.get(run_id))

    assert run is not None
    assert run.status == "coding"
    assert run.original_task == "完成微信小程序功能"
    assert run.workspace_dir == str(tmp_path / "life-note-miniapp")
    assert run.pending_child_ids == ["child-1"]


def test_terminal_run_rejects_late_state_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    async def scenario():
        store = RunStore()
        run = await store.create("channel", "thread", "message")
        await store.update(run.id, status="completed", outputs={"final": "ok"})
        before = (tmp_path / "feishu" / "runs" / f"{run.id}.json").read_text("utf-8")
        await store.patch_outputs(run.id, {"late": True})
        await store.record_runtime_progress(run.id, activity=True, tool="late-tool")
        await store.update(run.id, status="failed", error="late error")
        after = (tmp_path / "feishu" / "runs" / f"{run.id}.json").read_text("utf-8")
        return await store.get(run.id), before, after

    run, before, after = asyncio.run(scenario())
    assert run.status == "completed"
    assert run.outputs == {"final": "ok"}
    assert run.last_tool == ""
    assert before == after


def test_run_status_updates_keep_existing_card_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    async def scenario():
        store = RunStore()
        run = await store.create("channel", "thread", "message")
        await store.update(
            run.id,
            status="planning",
            outputs={"delivery": {"card_message_id": "om_one_card"}},
        )
        await store.update(
            run.id,
            status="waiting_children",
            outputs={"summary": "等待子任务"},
        )
        return await store.get(run.id)

    run = asyncio.run(scenario())
    assert run.outputs["delivery"]["card_message_id"] == "om_one_card"
    assert run.outputs["summary"] == "等待子任务"


def test_supervisor_uses_file_change_before_declaring_stall(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "stall_check_after", 0.01)
    monkeypatch.setattr(settings, "stall_confirmations", 2)

    class LiveProcess:
        returncode = None
        pid = 0

    async def scenario():
        supervisor = RunSupervisor()
        tracked = tmp_path / "result.txt"
        tracked.write_text("one", encoding="utf-8")
        await supervisor.register_process(
            run_id="run",
            invocation_id="invocation",
            process=LiveProcess(),
        )
        await supervisor.record_tool_call(
            "run", "invocation", "write", args={"path": str(tracked)}
        )
        await asyncio.sleep(0.02)
        tracked.write_text("two", encoding="utf-8")
        os.utime(tracked, None)
        state = supervisor._states[("run", "invocation")]
        state.last_progress_at -= 2
        state.last_activity_at -= 2
        changed = await supervisor.assess("run", "invocation")
        state.last_progress_at -= 2
        state.last_activity_at -= 2
        first_silent = await supervisor.assess("run", "invocation")
        second_silent = await supervisor.assess("run", "invocation")
        await supervisor.unregister_process("run", "invocation")
        return changed, first_silent, second_silent

    assert asyncio.run(scenario()) == ("working", "suspected_stall", "stalled")


def test_delegate_returns_child_number_without_running_child(monkeypatch):
    from api import internal
    from core import run_store as run_store_module
    from core import task_queue as task_queue_module
    from core.delegate_registry import delegate_registry

    class FakeStore:
        def __init__(self):
            self.child = None

        async def add_delegation(self, run_id, child):
            self.child = child
            return SimpleNamespace(id=run_id)

        async def update_delegation(self, *args, **kwargs):
            return None

    class FakeQueue:
        def __init__(self):
            self.queued = None

        async def enqueue_delegation(self, run_id, child_id):
            self.queued = (run_id, child_id)
            return True

    async def scenario():
        fake_store = FakeStore()
        fake_queue = FakeQueue()
        monkeypatch.setattr(run_store_module, "run_store", fake_store)
        monkeypatch.setattr(task_queue_module, "feishu_task_queue", fake_queue)
        monkeypatch.setattr(internal.registry, "get", lambda agent_id: DelegateTarget())
        token = await delegate_registry.issue(
            project_id=None,
            channel_id="channel",
            thread_id="thread",
            run_id="async-delegate-run",
            invocation_id="parent-1",
            caller_agent_id="coordinator",
            mcp_capabilities=["delegate"],
        )
        try:
            response = await internal.delegate(
                internal.DelegateRequest(
                    token=token, target="coder", task="实现页面", access="read"
                ),
                SimpleNamespace(client=SimpleNamespace(host="testclient")),
            )
        finally:
            await delegate_registry.revoke(token)
        return response, fake_store, fake_queue

    response, store, queue = asyncio.run(scenario())
    assert response.status == "queued"
    assert response.child_run_id == store.child.id
    assert store.child.parent_invocation_id == "parent-1"
    assert queue.queued == ("async-delegate-run", response.child_run_id)


def test_background_child_progress_is_forwarded_to_the_run_card(monkeypatch, tmp_path):
    from core import task_queue as task_queue_module
    from integrations.feishu import progress as progress_module
    from core.task_queue import DelegationTask, FeishuTaskQueue

    class Sender:
        def __init__(self):
            self.cards_sent = 0
            self.cards_updated = 0

        async def send_card(self, message_id, card, **kwargs):
            self.cards_sent += 1
            return "om_shared_card"

        async def update_card(self, card_message_id, card):
            self.cards_updated += 1
            return True

        async def reply(self, *args, **kwargs):
            return []

    class ChildOrchestrator:
        async def invoke_for_delegate(self, **kwargs):
            callback = kwargs["on_progress"]
            await callback(
                "stage", {"agent_id": "coder", "agent_name": "暹罗", "phase": "coding"}
            )
            await callback(
                "tool_call",
                {
                    "tool": "Write",
                    "args": {"file_path": "pages/detail.js"},
                    "agent_id": "coder",
                    "agent_name": "暹罗",
                    "phase": "coding",
                },
            )
            return "子任务完成", None

    async def scenario():
        store = RunStore()
        events = RunEventStore(tmp_path / "events")
        monkeypatch.setattr(task_queue_module, "run_store", store)
        monkeypatch.setattr(task_queue_module, "run_event_store", events)
        monkeypatch.setattr(progress_module, "run_event_store", events)
        monkeypatch.setattr(task_queue_module, "orchestrator", ChildOrchestrator())

        run = await store.create("channel", "thread", "message")
        await store.update(
            run.id,
                status="waiting_children",
                original_task="完成页面",
                workspace_dir=str(tmp_path),
                root_invocation_id="root-1",
        )
        await store.add_delegation(
            run.id,
            DelegationRecord(
                id="child-progress",
                parent_invocation_id="root-1",
                target_agent_id="coder",
                task="实现详情页",
                access="write",
            ),
        )

        queue = FeishuTaskQueue(sender=Sender())

        async def resolve_agent(project_id, agent_id):
            return DelegateTarget()

        monkeypatch.setattr(queue, "_resolve_entry_agent", resolve_agent)
        await queue._execute_delegation(
            DelegationTask(run_id=run.id, child_id="child-progress")
        )
        return await store.get_delegation(run.id, "child-progress"), await events.list(run.id), queue

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    child, events, queue = asyncio.run(scenario())
    assert child.status == "completed"
    assert any(event.agent_name == "暹罗" and "正在修改 pages/detail.js" in event.title for event in events)
    assert queue.sender.cards_sent == 1
    assert queue.sender.cards_updated >= 1


def test_durable_group_binding_overrides_stale_run_workspace(monkeypatch, tmp_path):
    from core import task_queue as task_queue_module
    from core.task_queue import FeishuTaskQueue
    from models.schemas import FeishuBinding

    data_dir = tmp_path / "data"
    workspace = tmp_path / "life-note-miniapp"
    workspace.mkdir()
    data_dir.mkdir()
    (data_dir / "workspace_roots.json").write_text(
        json.dumps([str(tmp_path)]), encoding="utf-8"
    )
    monkeypatch.setattr(settings, "data_dir", data_dir)

    class Mappings:
        async def list_bindings(self):
            return [FeishuBinding(
                id="binding",
                tenant_key="tenant",
                chat_id="chat",
                channel_id="channel",
                workspace_dir=str(workspace),
                project_id="project",
            )]

    async def scenario():
        store = RunStore()
        run = await store.create("channel", "thread", "message")
        await store.update(run.id, workspace_dir=str(tmp_path / "stale"))
        monkeypatch.setattr(task_queue_module, "run_store", store)
        monkeypatch.setattr(task_queue_module, "feishu_mapping_store", Mappings())
        queue = FeishuTaskQueue()
        resolved = await queue._pin_run_workspace(run)
        return resolved, await store.get(run.id)

    resolved, run = asyncio.run(scenario())
    assert resolved == str(workspace)
    assert run.workspace_dir == str(workspace)
