"""MCP delegate server: a stdio JSON-RPC tool exposed to Claude Code.

Claude Code spawns this script as an MCP child process (`--mcp-config`). It
exposes only the granted `delegate` and/or `report_progress` tools and forwards
their calls to the CatTogether backend over HTTP.

It is a stateless thin proxy: it knows nothing about projects or rosters. The
roster is injected into the agent's prompt by the orchestrator. This server
only relays `{token, target, task, access}` to `CT_DELEGATE_ENDPOINT/internal/delegate`.

Env (set per agent run via the .mcp.json config):
  CT_DELEGATE_TOKEN     — token identifying this agent's delegation context
  CT_DELEGATE_ENDPOINT  — backend base URL, e.g. http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os

import httpx
from mcp.server.mcpserver import MCPServer


mcp = MCPServer("cattogether")
_CAPABILITIES = {
    item.strip()
    for item in os.environ.get(
        "CT_MCP_CAPABILITIES", "delegate,report_progress"
    ).split(",")
    if item.strip()
}


def delegate(target: str, task: str, access: str = "read") -> str:
    """Pull another agent into the collaboration.

    Args:
        target: agent_id of the peer to pull (e.g. "coder", "researcher").
        task:   the subtask to hand to that agent.
        access: "read" or "write" — whether the peer may modify the workspace.

    Returns:
        The peer agent's text output, or an error message string.
    """
    token = os.environ.get("CT_DELEGATE_TOKEN", "")
    endpoint = os.environ.get("CT_DELEGATE_ENDPOINT", "http://127.0.0.1:8000").rstrip("/")
    if not token:
        return "delegate 未配置：缺少 CT_DELEGATE_TOKEN"
    url = f"{endpoint}/internal/delegate"
    try:
        # The backend now persists the work order and returns its child_run_id
        # immediately. This HTTP call no longer spans the child agent lifetime.
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                url,
                json={"token": token, "target": target, "task": task, "access": access},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return f"delegate 调用后端失败：{exc}"
    if data.get("error"):
        return f"delegate 失败：{data['error']}"
    return json.dumps(data, ensure_ascii=False)


def report_progress(message: str, kind: str = "checkpoint") -> str:
    """Report a short, user-visible work checkpoint to CatTogether.

    This is for actions and confirmed intermediate findings, not private chain
    of thought or raw tool output.
    """
    token = os.environ.get("CT_DELEGATE_TOKEN", "")
    endpoint = os.environ.get("CT_DELEGATE_ENDPOINT", "http://127.0.0.1:8000").rstrip("/")
    if not token:
        return "progress 未配置：缺少 CT_DELEGATE_TOKEN"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{endpoint}/internal/progress",
                json={"token": token, "message": message, "kind": kind},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return f"progress 调用后端失败：{exc}"
    if data.get("error"):
        return f"progress 失败：{data['error']}"
    return str(data.get("result") or "ok")


def memory_search(
    query: str,
    domain: str = "agent",
    scope_id: str = "",
    top_k: int = 5,
) -> str:
    """Search the memory system for relevant facts (read-only).

    Agents can call this to recall prior knowledge without going through the
    full context-build pipeline.  This is the Agent boundary reversal (2.1 R4):
    agents MAY directly call the Memory API as a tool, but only for reads.
    Writes still go through the Core Layer admission pipeline.

    Args:
        query:    The search query text (natural language or keywords).
        domain:   Memory domain — "user", "project", "task", or "agent".
                  ("workspace" accepted as alias for "project".)
        scope_id: The scope identifier within the domain (e.g. project_id,
                  user_id, task_id).  Empty string uses the agent's own scope.
        top_k:    Maximum number of results to return (1-20).

    Returns:
        JSON array of matching facts with text, score, kind, and metadata.
    """
    token = os.environ.get("CT_DELEGATE_TOKEN", "")
    endpoint = os.environ.get("CT_DELEGATE_ENDPOINT", "http://127.0.0.1:8000").rstrip("/")
    if not token:
        return "memory_search 未配置：缺少 CT_DELEGATE_TOKEN"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{endpoint}/internal/memory/search",
                json={
                    "token": token,
                    "query": query,
                    "domain": domain,
                    "scope_id": scope_id,
                    "top_k": top_k,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return f"memory_search 调用后端失败：{exc}"
    if data.get("error"):
        return f"memory_search 失败：{data['error']}"
    return json.dumps(data.get("results", []), ensure_ascii=False)


for _name, _func in {
    "delegate": delegate,
    "report_progress": report_progress,
    "memory_search": memory_search,
}.items():
    if _name in _CAPABILITIES:
        mcp.tool()(_func)


if __name__ == "__main__":
    mcp.run(transport="stdio")
