"""Wiki MCP server: lets agents read/write the project's Feishu Wiki.

Mounted alongside the delegate server in the same .mcp.json. Exposes:
  wiki_write(title, content) -> url      create a new Wiki node
  wiki_append(node_token, content)       append to an existing node
  wiki_read(node_token) -> content       read a node's content
  wiki_list() -> [{node_token, title, url}]   list nodes in the project space

All calls are relayed to the backend's /internal/wiki/* endpoints via HTTP,
using the same CT_DELEGATE_TOKEN that identifies the agent's project context.
"""
from __future__ import annotations

import os

import httpx
from mcp.server.mcpserver import MCPServer


mcp = MCPServer("cattogether-wiki")
_CAPABILITIES = {
    item.strip()
    for item in os.environ.get(
        "CT_MCP_CAPABILITIES",
        "wiki_write,wiki_append,wiki_read,wiki_list",
    ).split(",")
    if item.strip()
}


def _post(endpoint: str, action: str, **extra) -> str:
    token = os.environ.get("CT_DELEGATE_TOKEN", "")
    base = os.environ.get("CT_DELEGATE_ENDPOINT", "http://127.0.0.1:8000").rstrip("/")
    if not token:
        return "wiki 工具未配置：缺少 CT_DELEGATE_TOKEN"
    url = f"{base}/internal/wiki/{action}"
    body = {"token": token, **extra}
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return f"wiki 调用后端失败：{exc}"
    if data.get("error"):
        return f"wiki 失败：{data['error']}"
    return str(data.get("result") or "")


def wiki_write(title: str, content: str) -> str:
    """Create a new Feishu Wiki document in the project's space and return its URL.

    Use when the user asks to "整理成文档" / "写个文档" / "总结到文档".
    """
    return _post("", "write", title=title, content=content)


def wiki_append(node_token: str, content: str) -> str:
    """Append content to an existing Wiki node identified by node_token."""
    return _post("", "append", node_token=node_token, content=content)


def wiki_read(node_token: str) -> str:
    """Read the content of an existing Wiki node as markdown."""
    return _post("", "read", node_token=node_token)


def wiki_list() -> str:
    """List all Wiki nodes in the project's space (titles + URLs)."""
    return _post("", "list")


for _name, _func in {
    "wiki_write": wiki_write,
    "wiki_append": wiki_append,
    "wiki_read": wiki_read,
    "wiki_list": wiki_list,
}.items():
    if _name in _CAPABILITIES:
        mcp.tool()(_func)


if __name__ == "__main__":
    mcp.run(transport="stdio")
