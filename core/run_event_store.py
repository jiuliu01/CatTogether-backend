"""Append-only JSONL persistence for user-visible Run progress events."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from config import settings
from models.schemas import RunProgressEvent


class RunEventStore:
    def __init__(self, directory: Path | None = None) -> None:
        self._dir = directory or (settings.data_dir / "feishu" / "run_events")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._seqs: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def _path(self, run_id: str) -> Path:
        return self._dir / f"{run_id}.jsonl"

    def _last_seq(self, run_id: str) -> int:
        path = self._path(run_id)
        if not path.exists():
            return 0
        last = 0
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    last = max(last, int(json.loads(line).get("seq", 0)))
        except (OSError, ValueError, TypeError):
            return last
        return last

    async def append(self, event: RunProgressEvent) -> RunProgressEvent:
        async with self._lock:
            current = self._seqs.get(event.run_id)
            if current is None:
                current = self._last_seq(event.run_id)
            event.seq = current + 1
            self._seqs[event.run_id] = event.seq
            with self._path(event.run_id).open("a", encoding="utf-8") as fh:
                fh.write(event.model_dump_json() + "\n")
        return event

    async def list(self, run_id: str, limit: int | None = None) -> list[RunProgressEvent]:
        path = self._path(run_id)
        if not path.exists():
            return []
        events: list[RunProgressEvent] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(RunProgressEvent.model_validate_json(line))
        except (OSError, ValueError):
            return events[-limit:] if limit else events
        return events[-limit:] if limit else events


run_event_store = RunEventStore()
