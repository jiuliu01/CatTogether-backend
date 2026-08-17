"""Integration test: the reporter sends ONE card and patches it, not many texts."""
import asyncio
import uuid

from core.run_event_store import RunEventStore
from integrations.feishu.progress import FeishuProgressReporter


class FakeSender:
    def __init__(self):
        self.cards_sent = 0
        self.cards_updated = 0
        self.text_replies = []
        self._next_msg_id = 100
        self.last_card = None

    async def send_card(self, message_id, card, **kw):
        self.cards_sent += 1
        self.last_card = card
        self._next_msg_id += 1
        return f"om_card_{self._next_msg_id}"

    async def update_card(self, card_message_id, card):
        self.cards_updated += 1
        self.last_card = card
        return True

    async def reply(self, message_id, text, **kw):
        self.text_replies.append(text)
        return []


def test_reporter_sends_one_card_and_patches_it():
    async def run():
        sender = FakeSender()
        reporter = FeishuProgressReporter(sender, "om_1", "run_1", update_interval=0.0)

        await reporter.on_progress("stage", {"phase": "entry", "agent_name": "橘长"})
        await reporter.on_progress("text", {"delta": "查看目录…", "agent_name": "橘长"})
        await reporter.on_progress("text", {"delta": "找到了 main.py", "agent_name": "橘长"})
        await reporter.on_progress("stage", {"phase": "coding", "agent_name": "暹罗"})
        await reporter.on_progress("text", {"delta": "改了 api/rest.py", "agent_name": "暹罗"})
        reporter._card.set_short_result("已完成 /hello 接口。")
        await reporter.finalize()
        return sender

    sender = asyncio.run(run())
    assert sender.cards_sent == 1
    assert sender.cards_updated >= 2
    assert sender.text_replies == []


def test_reporter_reuses_existing_card_and_restores_parent_child_timeline(monkeypatch, tmp_path):
    from integrations.feishu import progress as progress_module

    event_store = RunEventStore(tmp_path / "events")
    monkeypatch.setattr(progress_module, "run_event_store", event_store)

    async def run():
        sender = FakeSender()
        run_id = f"run_resume_{uuid.uuid4().hex}"
        first = FeishuProgressReporter(
            sender, "om_1", run_id, update_interval=0.0,
            entry_agent_name="橘长",
        )
        await first.start(agent_id="coordinator")
        await first.on_progress(
            "text", {"delta": "父任务正在检查", "agent_name": "橘长", "phase": "entry"}
        )
        card_id = first.delivery_snapshot()["card_message_id"]
        await first.pause()

        resumed = FeishuProgressReporter(
            sender, "om_1", run_id, update_interval=0.0,
            entry_agent_name="橘长",
            existing_card_message_id=card_id,
            elapsed_seconds=30,
        )
        resumed.restore_events(await event_store.list(run_id))
        await resumed.activate()
        await resumed.on_progress(
            "stage", {"agent_id": "coder", "agent_name": "暹罗", "phase": "coding"}
        )
        await resumed.on_progress(
            "tool_call",
            {
                "tool": "Write",
                "args": {"file_path": "pages/home.js"},
                "agent_id": "coder",
                "agent_name": "暹罗",
                "phase": "coding",
            },
        )
        resumed._card.set_final_message("父子任务均已完成。")
        await resumed.finalize()
        return sender, resumed

    sender, reporter = asyncio.run(run())
    assert sender.cards_sent == 1
    assert sender.cards_updated >= 3
    process = reporter.process_snapshot()
    assert any(item["agent_name"] == "橘长" for item in process)
    assert any(item["agent_name"] == "暹罗" and "正在修改 pages/home.js" in item["text"] for item in process)
    assert "任务已完成" in sender.last_card["header"]["title"]["content"]


def test_reporter_falls_back_to_text_if_card_fails():
    async def run():
        sender = FakeSender()
        attempts = 0

        async def fail_send(*a, **kw):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("feishu down")
        sender.send_card = fail_send

        reporter = FeishuProgressReporter(
            sender,
            "om_1",
            "run_2",
            update_interval=0.0,
            entry_agent_name="橘长",
        )
        await reporter.on_progress("stage", {"phase": "entry", "agent_name": "橘长"})
        await reporter.on_progress("text", {"delta": "思考中", "agent_name": "橘长"})
        reporter._card.set_short_result("最终结果。")
        await reporter.finalize()
        return sender, attempts

    sender, attempts = asyncio.run(run())
    assert sender.cards_sent == 0
    assert attempts == 1
    assert any("⏳" in t and "思考中" not in t for t in sender.text_replies)
    assert any("最终结果" in t for t in sender.text_replies)
    assert any("回复角色：橘长" in t for t in sender.text_replies)


def test_reporter_can_use_text_mode_when_card_v2_is_disabled(monkeypatch):
    async def run():
        from integrations.feishu import progress as progress_module

        monkeypatch.setattr(progress_module.settings, "feishu_card_v2", False)
        sender = FakeSender()
        reporter = FeishuProgressReporter(
            sender, "om_1", "run_text_mode", update_interval=0.0,
            entry_agent_name="橘长",
        )
        await reporter.start(agent_id="coordinator")
        reporter._card.set_final_message("完整最终答案。")
        await reporter.finalize()
        return sender, reporter

    sender, reporter = asyncio.run(run())
    assert sender.cards_sent == 0
    assert any("完整最终答案" in text for text in sender.text_replies)
    assert reporter.delivery_snapshot()["state"] == "text_final_fallback"


def test_reporter_throttles_updates():
    async def run():
        sender = FakeSender()
        reporter = FeishuProgressReporter(sender, "om_1", "run_3", update_interval=100.0)

        await reporter.on_progress("stage", {"phase": "entry", "agent_name": "橘长"})
        for _ in range(20):
            await reporter.on_progress("text", {"delta": "x", "agent_name": "橘长"})
        reporter._card.set_short_result("done")
        await reporter.finalize()
        return sender

    sender = asyncio.run(run())
    assert sender.cards_sent == 1
    assert sender.cards_updated <= 2


def test_reporter_starts_immediately_and_emits_silent_heartbeat():
    async def run():
        sender = FakeSender()
        reporter = FeishuProgressReporter(
            sender,
            "om_1",
            "run_heartbeat",
            update_interval=0.0,
            heartbeat_interval=0.05,
            entry_agent_name="橘长",
        )
        await reporter.start(agent_id="coordinator")
        await asyncio.sleep(0.12)
        reporter._card.set_short_result("完成。")
        await reporter.finalize()
        return sender, reporter

    sender, reporter = asyncio.run(run())
    assert sender.cards_sent == 1
    assert sender.cards_updated >= 1
    assert reporter.delivery_snapshot()["event_count"] >= 2


def test_tool_action_updates_latest_progress_and_folded_timeline():
    async def run():
        sender = FakeSender()
        reporter = FeishuProgressReporter(
            sender, "om_1", "run_tool", update_interval=0.0,
            entry_agent_name="橘长",
        )
        await reporter.start(agent_id="coordinator")
        await reporter.on_progress("tool_call", {
            "tool": "Read",
            "args": {"file_path": "backend/main.py"},
            "agent_id": "coordinator",
            "agent_name": "橘长",
            "phase": "entry",
        })
        reporter._card.set_short_result("完成。")
        await reporter.finalize()
        return sender, reporter

    sender, reporter = asyncio.run(run())
    process = reporter.process_snapshot()
    assert "正在读取 backend/main.py" in process[0]["text"]
    panels = [e for e in sender.last_card["body"]["elements"] if e["tag"] == "collapsible_panel"]
    assert panels and "工作过程" in panels[0]["header"]["title"]["content"]


def test_duplicate_checkpoints_are_coalesced(monkeypatch):
    async def run():
        from integrations.feishu import progress as progress_module

        monkeypatch.setattr(progress_module.settings, "checkpoint_min_interval", 30)
        sender = FakeSender()
        reporter = FeishuProgressReporter(
            sender, "om_1", "run_checkpoint", update_interval=0.0,
            entry_agent_name="橘长",
        )
        payload = {
            "title": "已确认根因",
            "agent_name": "橘长",
            "phase": "entry",
        }
        await reporter.on_progress("checkpoint", payload)
        await reporter.on_progress("checkpoint", payload)
        reporter._card.set_final_message("完成。")
        await reporter.finalize()
        return reporter

    reporter = asyncio.run(run())
    matching = [
        stage["text"] for stage in reporter.process_snapshot()
        if "已确认根因" in stage["text"]
    ]
    assert len(matching) == 1
    assert matching[0].count("已确认根因") == 1
