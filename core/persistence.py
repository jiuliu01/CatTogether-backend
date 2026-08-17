"""Small persistence helpers shared by JSON-backed stores."""
from __future__ import annotations

import uuid
from pathlib import Path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace *path* only after the complete new content is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(text, encoding=encoding)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
