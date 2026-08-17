import asyncio
import pytest

from integrations.feishu import sender as sender_module
from integrations.feishu.sender import FeishuAPIError, FeishuSender


def test_split_long_text_without_losing_content():
    text = ("一段内容\n" * 1200).strip()
    chunks = FeishuSender._split_text(text, limit=500)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_update_card_uses_patch_without_msg_type(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"code": 0}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def patch(self, url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse()

        async def post(self, url, **kwargs):
            raise AssertionError("card update must use PATCH")

        async def put(self, url, **kwargs):
            raise AssertionError("card update must use PATCH")

    async def scenario():
        sender = FeishuSender()

        async def token():
            return "tenant-token"

        sender._access_token = token
        monkeypatch.setattr(sender_module.httpx, "AsyncClient", FakeClient)
        return await sender.update_card("om_card", {"schema": "2.0"})

    assert asyncio.run(scenario()) is True
    assert calls and calls[0][0].endswith("/open-apis/im/v1/messages/om_card")
    assert "msg_type" not in calls[0][1]["json"]


def test_http_400_keeps_feishu_diagnostic_and_is_not_retried(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 400
        headers = {"x-request-id": "req-400"}

        def json(self):
            return {"code": 230001, "msg": "invalid card content"}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse()

    async def scenario():
        sender = FeishuSender()

        async def token():
            return "tenant-token"

        sender._access_token = token
        monkeypatch.setattr(sender_module.httpx, "AsyncClient", FakeClient)
        await sender._post_with_retry("https://example.test/card", {"x": 1})

    with pytest.raises(FeishuAPIError) as exc:
        asyncio.run(scenario())

    assert len(calls) == 1
    assert "code=230001" in str(exc.value)
    assert "invalid card content" in str(exc.value)
    assert "request_id=req-400" in str(exc.value)


def test_update_card_retries_transient_5xx(monkeypatch):
    statuses = [503, 200]

    class FakeResponse:
        headers = {}

        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            return {"code": 0, "msg": "ok"}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def patch(self, url, **kwargs):
            return FakeResponse(statuses.pop(0))

        async def post(self, url, **kwargs):
            raise AssertionError("not expected")

        async def put(self, url, **kwargs):
            raise AssertionError("not expected")

    async def scenario():
        sender = FeishuSender()

        async def token():
            return "tenant-token"

        async def no_sleep(_seconds):
            return None

        sender._access_token = token
        monkeypatch.setattr(sender_module.httpx, "AsyncClient", FakeClient)
        monkeypatch.setattr(sender_module.asyncio, "sleep", no_sleep)
        return await sender.update_card("om_card", {"schema": "2.0"})

    assert asyncio.run(scenario()) is True
    assert statuses == []


def test_card_validation_rejects_legacy_v1_component():
    card = {
        "schema": "2.0",
        "body": {"elements": [{"tag": "note", "elements": []}]},
    }

    with pytest.raises(FeishuAPIError) as exc:
        FeishuSender._validate_card_payload(card)

    assert "legacy tag: note" in str(exc.value)
