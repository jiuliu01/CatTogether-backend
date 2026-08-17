import asyncio

from core.run_event_store import RunEventStore
from models.schemas import RunProgressEvent


def test_run_events_append_as_ordered_jsonl(tmp_path):
    async def scenario():
        store = RunEventStore(tmp_path / "events")
        first = await store.append(RunProgressEvent(
            run_id="run1", kind="stage", title="开始处理",
        ))
        second = await store.append(RunProgressEvent(
            run_id="run1", kind="action", title="正在读取文件",
        ))
        return first, second, await store.list("run1")

    first, second, events = asyncio.run(scenario())
    assert (first.seq, second.seq) == (1, 2)
    assert [event.title for event in events] == ["开始处理", "正在读取文件"]
