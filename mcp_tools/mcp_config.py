"""Generate per-run .mcp.json configs so each Claude Code subprocess gets its
own delegation token pointing at the backend.

Claude Code reads a config like:
  {
    "mcpServers": {
      "cattogether": {
        "command": "<python>",
        "args": ["<abs path to delegate_server.py>"],
        "env": {"CT_DELEGATE_TOKEN": "...", "CT_DELEGATE_ENDPOINT": "..."}
      }
    }
  }

We write one temp file per run under CT_MCP_CONFIG_DIR and pass its path to
`claude --mcp-config`. The CLI executor cleans it up after the run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from config import settings


def _server_script_path() -> str:
    # mcp_tools/delegate_server.py lives next to this file.
    return str(Path(__file__).resolve().parent / "delegate_server.py")


def _wiki_server_script_path() -> str:
    return str(Path(__file__).resolve().parent / "wiki_server.py")


def _python_path() -> str:
    # Use the same interpreter that runs the backend (the venv with mcp installed).
    return sys.executable


def _config_dir() -> Path:
    d = Path(settings.mcp_config_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_mcp_config(token: str, capabilities: list[str] | tuple[str, ...] | None = None) -> str:
    """Write a .mcp.json for this run's token; return its absolute path.

    Collaboration is mounted by default. Wiki is mounted only when this run
    was granted at least one wiki_* capability.
    """
    caps = set(
        ("delegate", "report_progress") if capabilities is None else capabilities
    )
    servers: dict[str, dict] = {}
    if caps & {"delegate", "report_progress", "memory_search"}:
        servers["cattogether"] = {
            "command": _python_path(),
            "args": [_server_script_path()],
            "env": {
                "CT_DELEGATE_TOKEN": token,
                "CT_DELEGATE_ENDPOINT": settings.delegate_endpoint,
                "CT_DELEGATE_TIMEOUT": str(settings.delegate_timeout),
                "CT_MCP_CAPABILITIES": ",".join(sorted(caps & {"delegate", "report_progress", "memory_search"})),
            },
        }
    wiki_caps = sorted(cap for cap in caps if cap.startswith("wiki_"))
    if wiki_caps:
        servers["cattogether-wiki"] = {
            "command": _python_path(),
            "args": [_wiki_server_script_path()],
            "env": {
                "CT_DELEGATE_TOKEN": token,
                "CT_DELEGATE_ENDPOINT": settings.delegate_endpoint,
                "CT_MCP_CAPABILITIES": ",".join(wiki_caps),
            },
        }
    payload = {
        "mcpServers": servers,
    }
    path = _config_dir() / f"{token}.mcp.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def remove_mcp_config(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
