"""Verify the wiki MCP server exposes working tools and relays to the backend."""
import asyncio

import httpx

from mcp_tools import wiki_server


def test_wiki_tools_registered():
    tools = asyncio.run(wiki_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert {"wiki_write", "wiki_append", "wiki_read", "wiki_list"} <= names


def test_wiki_write_relays_to_backend(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": "https://tenant.feishu.cn/wiki/nt_new", "error": None}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResp()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.setenv("CT_DELEGATE_TOKEN", "tok123")
    monkeypatch.setenv("CT_DELEGATE_ENDPOINT", "http://127.0.0.1:8000")

    url = wiki_server.wiki_write(title="测试文档", content="## 内容")

    assert captured["url"].endswith("/internal/wiki/write")
    assert captured["json"]["token"] == "tok123"
    assert captured["json"]["title"] == "测试文档"
    assert "https://tenant.feishu.cn/wiki/nt_new" in url


def test_wiki_list_relays_to_backend(monkeypatch):
    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": "- 文档1：url1\n- 文档2：url2", "error": None}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            return FakeResp()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.setenv("CT_DELEGATE_TOKEN", "tok")
    monkeypatch.setenv("CT_DELEGATE_ENDPOINT", "http://127.0.0.1:8000")

    out = wiki_server.wiki_list()
    assert "文档1" in out and "url1" in out


def test_wiki_tool_requires_token(monkeypatch):
    monkeypatch.delenv("CT_DELEGATE_TOKEN", raising=False)
    out = wiki_server.wiki_write(title="x", content="y")
    assert "未配置" in out


def test_wiki_tool_reports_backend_error(monkeypatch):
    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": "", "error": "invalid or expired token"}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            return FakeResp()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.setenv("CT_DELEGATE_TOKEN", "tok")
    monkeypatch.setenv("CT_DELEGATE_ENDPOINT", "http://127.0.0.1:8000")

    out = wiki_server.wiki_read(node_token="nt_x")
    assert "失败" in out
