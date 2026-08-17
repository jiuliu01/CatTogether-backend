"""Workspace management for file-operating agents (Codex / Claude Code).

Each channel gets a working directory under data/workspaces/<channel_id>/.
The user may also pin a project root via REST so CLI agents operate on a real
codebase instead of a scratch dir.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from config import settings
from core.identifiers import validate_storage_id


class WorkspaceManager:
    def __init__(self) -> None:
        self._pinned: dict[str, str] = {}  # channel_id -> absolute path

    def base_dir(self) -> Path:
        d = settings.data_dir / "workspaces"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_dir(self, channel_id: str) -> str:
        safe_channel_id = validate_storage_id(channel_id, "channel_id")
        pinned = self._pinned.get(channel_id)
        if pinned:
            Path(pinned).mkdir(parents=True, exist_ok=True)
            return pinned
        d = self.base_dir() / safe_channel_id
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    def allowed_roots(self) -> list[Path]:
        configured = [Path(p).expanduser().resolve() for p in settings.allowed_workspace_roots]
        if configured:
            return configured
        # A local durable allow-list is useful for desktop deployments where
        # the service is not launched from a shell that exports environment
        # variables. It remains explicit and narrower than allowing a drive.
        allowlist = settings.data_dir / "workspace_roots.json"
        try:
            values = json.loads(allowlist.read_text(encoding="utf-8"))
            stored = [
                Path(value).expanduser().resolve()
                for value in values
                if isinstance(value, str) and value.strip()
            ]
            if stored:
                return stored
        except (OSError, ValueError, TypeError):
            pass
        return [settings.project_root.resolve(), settings.data_dir.resolve()]

    def validate_path(self, path: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not any(resolved == root or root in resolved.parents for root in self.allowed_roots()):
            roots = ", ".join(str(root) for root in self.allowed_roots())
            raise ValueError(f"workspace must be inside an allowed root: {roots}")
        return resolved

    def pin(self, channel_id: str, path: str) -> None:
        self._pinned[channel_id] = str(self.validate_path(path))

    def unpin(self, channel_id: str) -> None:
        self._pinned.pop(channel_id, None)

    def clear(self, channel_id: str) -> None:
        safe_channel_id = validate_storage_id(channel_id, "channel_id")
        d = self.base_dir() / safe_channel_id
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


workspace_manager = WorkspaceManager()
