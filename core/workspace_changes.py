"""Detect real file changes made inside a bound workspace."""
from __future__ import annotations

import os
from pathlib import Path


FileState = tuple[int, int]
WorkspaceSnapshot = dict[str, FileState]

_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def snapshot_workspace(workspace_dir: str) -> WorkspaceSnapshot | None:
    """Return a lightweight snapshot, or None when the workspace is unavailable."""
    try:
        root = Path(workspace_dir).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not root.is_dir():
        return None

    snapshot: WorkspaceSnapshot = {}
    try:
        for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            dirnames[:] = [name for name in dirnames if name not in _IGNORED_DIRS]
            current_path = Path(current)
            for filename in filenames:
                path = current_path / filename
                try:
                    stat = path.stat()
                    relative = path.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                snapshot[relative] = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None
    return snapshot


def changed_workspace_paths(
    workspace_dir: str,
    before: WorkspaceSnapshot | None,
) -> list[str]:
    """List files created, removed, or modified since ``before``."""
    if before is None:
        return []
    after = snapshot_workspace(workspace_dir)
    if after is None:
        return []
    paths = set(before) | set(after)
    return sorted(path for path in paths if before.get(path) != after.get(path))

