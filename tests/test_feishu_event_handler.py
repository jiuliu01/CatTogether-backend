import asyncio

from integrations.feishu.event_handler import FeishuEventHandler
from tests.test_feishu_parser import message_event


class FakeMappings:
    def __init__(self):
        self.seen = set()

    async def mark_processed(self, message_id):
        if message_id in self.seen:
            return False
        self.seen.add(message_id)
        return True


class FakeTasks:
    def __init__(self):
        self.messages = []

    async def enqueue(self, inbound):
        self.messages.append(inbound)


class FakeSender:
    def __init__(self):
        self.replies = []

    async def reply(self, message_id, text, **kwargs):
        self.replies.append((message_id, text, kwargs))
        return []


def test_event_is_queued_once():
    async def scenario():
        mappings = FakeMappings()
        tasks = FakeTasks()
        handler = FeishuEventHandler(
            mappings=mappings,
            tasks=tasks,
            sender=FakeSender(),
        )
        payload = message_event()

        assert await handler.handle_raw(payload) == "queued"
        assert await handler.handle_raw(payload) == "duplicate"
        assert len(tasks.messages) == 1
        assert tasks.messages[0].content == "帮我规划这个项目"

    asyncio.run(scenario())
