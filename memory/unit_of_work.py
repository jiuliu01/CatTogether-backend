"""MemoryUnitOfWork — atomic write transaction for Memory 2.1 (§7).

A single ``BEGIN IMMEDIATE`` transaction that commits all side-effects of a
write operation atomically:

    Fact + Entity + EntityMention + FactRelation + embedding
    + Event status + Audit + Outbox

If any step fails, the entire transaction rolls back — no half-committed
records, no orphaned embeddings, no audit entries without facts.

Usage::

    uow = MemoryUnitOfWork(db)
    uow.begin()
    try:
        uow.add_fact(fact)
        uow.add_entity(entity)
        uow.add_mention(mention)
        uow.add_relation(relation)
        uow.add_embedding(fact_id, vector)
        uow.mark_event_processed(event_id)
        uow.add_audit(...)
        uow.add_outbox(...)
        uow.commit()
    except Exception:
        uow.rollback()
        raise

``BEGIN IMMEDIATE`` acquires a write lock at start (rather than at first
write) to avoid upgrade-deadlock with concurrent readers that later write.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Sequence

from memory.db import MemoryDB, memory_db


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryUnitOfWork:
    """Single-connection atomic write unit for the nine-stage pipeline.

    All writes go through one ``sqlite3.Connection`` inside one
    ``BEGIN IMMEDIATE`` transaction.  Call ``commit()`` to persist everything
    or ``rollback()`` to discard everything.
    """

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db
        self._conn: sqlite3.Connection | None = None
        self._active = False

    # ----- Transaction control -----

    def begin(self) -> sqlite3.Connection:
        """Begin a ``BEGIN IMMEDIATE`` transaction."""
        if self._active:
            raise RuntimeError("MemoryUnitOfWork already active")
        self._conn = self._db.connect()
        self._conn.execute("BEGIN IMMEDIATE")
        self._active = True
        return self._conn

    def commit(self) -> None:
        """Commit the transaction."""
        if not self._active or self._conn is None:
            raise RuntimeError("MemoryUnitOfWork not active")
        self._conn.execute("COMMIT")
        self._active = False
        self._conn = None

    def rollback(self) -> None:
        """Rollback the transaction (safe to call if not active)."""
        if not self._active or self._conn is None:
            return
        try:
            self._conn.execute("ROLLBACK")
        finally:
            self._active = False
            self._conn = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def conn(self) -> sqlite3.Connection:
        if not self._active or self._conn is None:
            raise RuntimeError("MemoryUnitOfWork not active — call begin() first")
        return self._conn

    # ----- Write operations (all within the transaction) -----

    def add_fact(self, fact: dict[str, Any]) -> str:
        """Insert a Fact row.  *fact* is a dict of column → value."""
        cols = list(fact.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_list = ", ".join(cols)
        sql = f"INSERT INTO facts ({col_list}) VALUES ({placeholders})"
        self.conn.execute(sql, tuple(fact[c] for c in cols))
        return fact["id"]

    def add_entity(self, entity: dict[str, Any]) -> str:
        """Insert an Entity row (or update last_seen_at if exists)."""
        self.conn.execute(
            """INSERT INTO entities (id, tenant_id, canonical_name, entity_type,
                  last_seen_at, merged_into_id)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(tenant_id, id) DO UPDATE SET last_seen_at=excluded.last_seen_at""",
            (
                entity["id"], entity["tenant_id"], entity["canonical_name"],
                entity.get("entity_type", "unknown"), entity.get("last_seen_at", _utc_now_iso()),
                entity.get("merged_into_id"),
            ),
        )
        return entity["id"]

    def add_mention(self, mention: dict[str, Any]) -> None:
        """Insert a fact_entity_mention row."""
        self.conn.execute(
            """INSERT INTO fact_entity_mentions
               (tenant_id, fact_id, entity_id, role, mention_text, linking_confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                mention["tenant_id"], mention["fact_id"], mention.get("entity_id"),
                mention.get("role", "mentions"), mention.get("mention_text", ""),
                mention.get("linking_confidence", 1.0),
            ),
        )

    def add_relation(self, relation: dict[str, Any]) -> None:
        """Insert a fact_relation row (idempotent via PK conflict)."""
        self.conn.execute(
            """INSERT OR IGNORE INTO fact_relations
               (tenant_id, src_fact_id, dst_fact_id, relation, confidence, created_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                relation["tenant_id"], relation["src_fact_id"], relation["dst_fact_id"],
                relation["relation"], relation.get("confidence", 1.0),
                relation.get("created_by", "system"), relation.get("created_at", _utc_now_iso()),
            ),
        )

    def add_embedding(self, fact_id: str, tenant_id: str, vector: Sequence[float]) -> None:
        """Insert or replace an embedding row in the vec0 virtual table."""
        vec_blob = sqlite3.Vector(vector)  # type: ignore[attr-defined]
        self.conn.execute(
            "INSERT OR REPLACE INTO fact_embeddings (fact_id, tenant_id, embedding) VALUES (?, ?, ?)",
            (fact_id, tenant_id, vec_blob),
        )

    def mark_event_status(self, event_id: str, status: str) -> None:
        """Update an event's status (pending → processed/failed/skipped)."""
        self.conn.execute(
            "UPDATE events SET status=?, updated_at=? WHERE id=?",
            (status, _utc_now_iso(), event_id),
        )

    def add_audit(
        self,
        *,
        tenant_id: str,
        action: str,
        actor: str,
        fact_id: str | None = None,
        domain: str | None = None,
        reason: str | None = None,
    ) -> str:
        """Insert an audit log entry (no fact text, ever)."""
        import uuid
        audit_id = f"aud_{uuid.uuid4().hex[:16]}"
        self.conn.execute(
            """INSERT INTO memory_audit_log
               (id, tenant_id, action, actor, fact_id, domain, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (audit_id, tenant_id, action, actor, fact_id, domain, reason, _utc_now_iso()),
        )
        return audit_id

    def add_outbox(
        self,
        *,
        tenant_id: str,
        event_type: str,
        payload: dict[str, Any],
        fact_id: str | None = None,
    ) -> str:
        """Insert a transaction outbox entry for downstream consumers.

        ``event_type`` maps to the ``op`` column (insert/update/delete/reindex).
        A NULL ``processed_at`` means pending.
        """
        import uuid
        outbox_id = f"obx_{uuid.uuid4().hex[:16]}"
        self.conn.execute(
            """INSERT INTO transaction_outbox
               (id, tenant_id, fact_id, op, payload, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (outbox_id, tenant_id, fact_id, event_type,
             json.dumps(payload, ensure_ascii=False), _utc_now_iso()),
        )
        return outbox_id

    # ----- Context manager support -----

    def __enter__(self) -> "MemoryUnitOfWork":
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.rollback()
        else:
            try:
                self.commit()
            except Exception:
                self.rollback()
                raise
