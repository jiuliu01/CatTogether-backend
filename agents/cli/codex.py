"""Codex CLI adapter.

Command shape:
  codex exec --json --sandbox <read-only|workspace-write>
    --ask-for-approval never --ephemeral --skip-git-repo-check -
(prompt via stdin, JSON events on stdout).

API-key auth: OPENAI_API_KEY inherited via env (set in cli_base._build_env).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import AsyncIterator

from agents.base import InvokeContext, RawEvent
from agents.cli.cli_base import CLIBaseAgent
from config import settings


class CodexAgent(CLIBaseAgent):
    def __init__(self, agent_id: str = "codex", name: str = "Codex", description: str = "OpenAI Codex CLI agent") -> None:
        super().__init__(agent_id, name, settings.codex_bin, description)

    def _bin_name(self) -> str:
        return "codex"

    def resolve_bin(self) -> str | None:
        if self.bin_path and os.path.isfile(self.bin_path):
            return self.bin_path

        # The Microsoft Store app execution alias may resolve into WindowsApps
        # but still reject direct subprocess execution. Prefer the real CLI
        # shipped with the Codex desktop app when available.
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            bin_root = Path(local_app_data) / "OpenAI" / "Codex" / "bin"
            candidates = sorted(
                bin_root.glob("*/codex.exe"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            ) if bin_root.exists() else []
            if candidates:
                return str(candidates[0])

        plugin_cli = Path.home() / ".codex" / "plugins" / ".plugin-appserver" / "codex.exe"
        if plugin_cli.is_file():
            return str(plugin_cli)

        resolved = super().resolve_bin()
        if resolved and "WindowsApps" not in resolved:
            return resolved
        return None

    def build_command(self, ctx: InvokeContext) -> list[str]:
        bin = self.resolve_bin() or "codex"
        sandbox = "workspace-write" if ctx.workspace_access == "write" else "read-only"
        return [
            bin,
            "--sandbox", sandbox,
            "--ask-for-approval", "never",
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "-",  # read prompt from stdin
        ]

    def _build_env(self) -> dict[str, str]:
        env = super()._build_env()
        if settings.openai_api_key:
            # CODEX_API_KEY is scoped to `codex exec`. Avoid also exposing the
            # generic OPENAI_API_KEY to model-generated child commands.
            env.pop("OPENAI_API_KEY", None)
            env["CODEX_API_KEY"] = settings.openai_api_key
        return env

    async def parse_chunk(self, raw: str, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        line = raw.strip()
        if not line:
            return
        # Codex emits JSON-stream objects; try to parse.
        try:
            obj = json.loads(line)
        except Exception:
            # not JSON: pass through as text
            yield ("text_delta", {"delta": raw})
            return
        t = obj.get("type")
        if t == "item.completed":
            item = obj.get("item") or {}
            item_type = item.get("type")
            if item_type == "agent_message":
                text = item.get("text") or ""
                if text:
                    yield ("text_delta", {"delta": text})
            elif item_type == "command_execution":
                command = item.get("command") or ""
                yield ("tool_call", {"tool": "command", "args": {"command": command}})
                output = item.get("aggregated_output") or item.get("output") or ""
                if output:
                    yield ("tool_result", {"tool": "command", "result": output})
            elif item_type == "file_change":
                yield ("tool_result", {"tool": "file_change", "result": item})
        elif t == "turn.failed":
            error = obj.get("error") or obj.get("message") or obj
            if isinstance(error, dict):
                error = error.get("message") or str(error)
            yield ("error", {"message": str(error)})
        elif t == "error":
            message = str(obj.get("message") or obj)
            if message.lower().startswith("reconnecting"):
                yield ("status", {"state": "running", "detail": message})
            else:
                yield ("error", {"message": message})
        elif t == "message" or t == "agent_message":
            text = obj.get("content") or obj.get("text") or ""
            if text:
                yield ("text_delta", {"delta": text})
        elif t == "command_execution":
            yield ("tool_call", {"tool": obj.get("name", "command"), "args": obj.get("args", {})})
        elif t == "command_execution_output" or t == "command_output":
            yield ("tool_result", {"tool": obj.get("name", "command"), "result": obj.get("output", "")})
        else:
            # unknown JSON event: surface as text if it carries content
            text = obj.get("content") or obj.get("text")
            if text:
                yield ("text_delta", {"delta": text})
