import asyncio
import threading

import lark_oapi as lark

from config import settings
from integrations.feishu.client import FeishuLongConnectionClient


def test_sdk_gets_a_dedicated_thread_event_loop(monkeypatch):
    started = threading.Event()
    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self._conn = object()
            self._auto_reconnect = True

        def start(self):
            captured["loop"] = lark.ws.client.loop
            started.set()

    monkeypatch.setattr(lark.ws, "Client", FakeClient)
    monkeypatch.setattr(settings, "feishu_app_id", "cli_test")
    monkeypatch.setattr(settings, "feishu_app_secret", "secret_test")
    monkeypatch.setattr(settings, "feishu_connection_mode", "long_connection")

    async def scenario():
        uvicorn_loop = asyncio.get_running_loop()
        client = FeishuLongConnectionClient()
        assert client.start(uvicorn_loop) is True
        assert await asyncio.to_thread(started.wait, 2)
        if client._thread:
            await asyncio.to_thread(client._thread.join, 2)
        assert captured["loop"] is not uvicorn_loop
        assert captured["loop"].is_closed()

    asyncio.run(scenario())
