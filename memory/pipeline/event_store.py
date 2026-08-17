"""Stage 1: Event Store — persist events with status=pending (§7.1).

The event store is the durable entry point of the write pipeline.  Every
write request must first be persisted as a ``pending`` event before any
processing begins.  This guarantees recoverability: if the process crashes
after the event is accepted but before processing completes, the event can
be replayed on restart.

Idempotency: ``put`` uses ``INSERT OR IGNORE`` on ``(tenant_id, event_id)``,
so re-submitting the same event is a no-op.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from memory.db import MemoryDB, memory_db
from memory.models import Event


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventStore:
    """Durable event storage for the write pipeline (§7.1 stage 1)."""

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db

    def put(self, event: Event) -> str:
        """Persist *event* with ``status='pending'``.

        Idempotent: re-putting the same ``(tenant_id, event_id)`` is a no-op.
        Returns the event_id.
        """
        conn = self._db.connect()
        conn.execute(
            """INSERT OR IGNORE INTO events
               (event_id, tenant_id, event_type, actor, scope_hint, payload,
                allowed_domains, created_at, processed_at, processing_stage,
                attempt_count, status, error, fact_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, 'pending', NULL, '[]')""",
            (
                event.event_id,
                event.tenant_id,
                event.event_type,
                event.actor.model_dump_json(),
                event.scope_hint,  # str | None, stored as-is
                json.dumps(event.payload, ensure_ascii=False),
                json.dumps(event.allowed_domains, ensure_ascii=False),
                event.created_at.isoformat() if event.created_at else _utc_now_iso(),
            ),
        )
        return event.event_id

    def is_processed(self, event_id: str, tenant_id: str) -> bool:
        """Check if an event has already been processed (idempotency)."""
        row = self._db.query_one(
            "SELECT status FROM events WHERE event_id = ? AND tenant_id = ?",
            (event_id, tenant_id),
        )
        return row is not None and row["status"] in ("processed", "skipped")

    def mark_status(
        self,
        event_id: str,
        tenant_id: str,
        status: str,
        *,
        error: str | None = None,
        fact_ids: list[str] | None = None,
    ) -> None:
        """Update an event's status (pending → processed/failed/skipped)."""
        self._db.execute(
            """UPDATE events
               SET status = ?, error = ?, fact_ids = ?,
                   processed_at = ?, attempt_count = attempt_count + 1
               WHERE event_id = ? AND tenant_id = ?""",
            (
                status,
                error,
                json.dumps(fact_ids or [], ensure_ascii=False),
                _utc_now_iso(),
                event_id,
                tenant_id,
            ),
        )

    def mark_stage(
        self,
        event_id: str,
        tenant_id: str,
        stage: str,
    ) -> None:
        """Update the processing checkpoint (current stage)."""
        self._db.execute(
            "UPDATE events SET processing_stage = ? WHERE event_id = ? AND tenant_id = ?",
            (stage, event_id, tenant_id),
        )

    def get_pending(
        self,
        tenant_id: str,
        limit: int = 100,
    ) -> list[Event]:
        """Fetch pending events for recovery/replay, oldest first."""
        rows = self._db.query_all(
            """SELECT * FROM events
               WHERE tenant_id = ? AND status = 'pending'
               ORDER BY created_at ASC LIMIT ?""",
            (tenant_id, limit),
        )
        return [self._row_to_event(r) for r in rows]

    def _row_to_event(self, row: Any) -> Event:
        """Reconstruct an Event from a database row."""
        from memory.models import Actor

        actor_data = json.loads(row["actor"]) if row["actor"] else {}
        # scope_hint is stored as a plain string (or None).
        scope_hint = row["scope_hint"] if row["scope_hint"] else None

        return Event(
            event_id=row["event_id"],
            tenant_id=row["tenant_id"],
            event_type=row["event_type"],
            actor=Actor(**actor_data),
            scope_hint=scope_hint,
            payload=json.loads(row["payload"]) if row["payload"] else {},
            allowed_domains=json.loads(row["allowed_domains"]) if row["allowed_domains"] else [],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(timezone.utc),
            processed_at=datetime.fromisoformat(row["processed_at"]) if row["processed_at"] else None,
            processing_stage=row["processing_stage"],
            attempt_count=row["attempt_count"],
            status=row["status"],
            error=row["error"],
            fact_ids=json.loads(row["fact_ids"]) if row["fact_ids"] else [],
        )
