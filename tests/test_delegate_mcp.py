"""Verify the MCP delegate server exposes a working `delegate` tool.

We spin up the FastMCP/MCPServer in-process and call the tool directly, with a
mocked backend response, to confirm: the tool is registered, its schema is
correct, and it relays {token, target, task, access} to the backend and returns
the result string.
"""
import asyncio

import httpx

from mcp_tools import delegate_server


def test_delegate_tool_is_registered():
    tools = asyncio.run(delegate_server.mcp.list_tools())
    assert {tool.name for tool in tools} == {"delegate", "report_progress"}
    tool = next(tool for tool in tools if tool.name == "delegate")
    assert tool.name == "delegate"
    props = set(tool.input_schema.get("properties", {}).keys())
    assert {"target", "task", "access"} <= props


def test_delegate_tool_relays_to_backend(monkeypatch):
    """The tool POSTs to /internal/delegate and returns the result."""
    captured = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": "[暹罗] done: 实现 /hello", "error": None}

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

    out = delegate_server.delegate(target="coder", task="实现 /hello", access="write")

    assert captured["url"] == "http://127.0.0.1:8000/internal/delegate"
    assert captured["json"]["token"] == "tok123"
    assert captured["json"]["target"] == "coder"
    assert captured["json"]["access"] == "write"
    assert "[暹罗]" in out


def test_delegate_tool_reports_backend_error(monkeypatch):
    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": "", "error": "委派深度超过上限"}

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

    out = delegate_server.delegate(target="x", task="y")
    assert "深度" in out


def test_delegate_tool_requires_token(monkeypatch):
    monkeypatch.delenv("CT_DELEGATE_TOKEN", raising=False)
    out = delegate_server.delegate(target="x", task="y")
    assert "未配置" in out
