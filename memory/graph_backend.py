"""Graph edge projection for Memory 2.0 (§5 库3, §6 环4 async, Stage 3).

Projects entity/relation edges from the ``facts`` main table into edge tables
for graph traversal:

- ``fact_entity_mentions`` — fact ↔ entity (populated by EntityLinker)
- ``fact_relations`` — fact ↔ fact (supersedes/supports/contradicts/derived_from)

This module provides the ``GraphBackend`` SQLite implementation and the
``GraphProjector`` that runs async projection from the transaction outbox.

The projection is idempotent: re-projecting the same fact produces the same
edges.  Edges are never mutated in-place; new edges are always INSERTed
(ADD-only, R4).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from memory.db import MemoryDB, memory_db
from memory.models import FactRelation, FactRelationType, GraphBackend
from memory.scope import MemoryScope


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteGraphBackend(GraphBackend):
    """Entity/relation edge storage and traversal backed by SQLite edge tables."""

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db

    @property
    def db(self) -> MemoryDB:
        return self._db

    # ----- Entity mention edges -----

    def upsert_entity_mention_sync(
        self,
        fact_id: str,
        entity_id: str,
        role: str,
        tenant_id: str,
    ) -> None:
        """Insert a fact↔entity mention edge (idempotent)."""
        # Map v3 roles to v4: context → object.
        v4_role = "object" if role == "context" else role
        conn = self._db.connect()
        conn.execute(
            """INSERT OR IGNORE INTO fact_entity_mentions
               (tenant_id, fact_id, entity_id, mention_text, role, linking_confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (tenant_id, fact_id, entity_id, "", v4_role, 0.8),
        )

    # ----- Fact relation edges -----

    def upsert_fact_relation_sync(
        self,
        src_fact_id: str,
        dst_fact_id: str,
        relation: FactRelationType,
        tenant_id: str,
        weight: float = 1.0,
    ) -> FactRelation:
        """Insert a fact↔fact relation edge (ADD-only, R4)."""
        now = _utc_now_iso()
        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            conn.execute(
                """INSERT OR IGNORE INTO fact_relations
                   (tenant_id, src_fact_id, dst_fact_id, relation, confidence, created_by, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (tenant_id, src_fact_id, dst_fact_id, relation, weight, "system", now),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return FactRelation(
            tenant_id=tenant_id,
            src_fact_id=src_fact_id,
            dst_fact_id=dst_fact_id,
            relation=relation,
            confidence=weight,
            created_by="system",
        )

    # ----- Graph traversal -----

    def entity_neighbors_sync(
        self,
        entity_id: str,
        tenant_id: str,
        max_hops: int = 2,
    ) -> list[str]:
        """BFS: find all fact_ids reachable from *entity_id* within *max_hops*.

        Hop 1: facts directly mentioning the entity.
        Hop 2: facts related to hop-1 facts via fact_relations.
        """
        conn = self._db.connect()
        visited_facts: set[str] = set()
        current_entities: set[str] = {entity_id}

        for hop in range(max_hops):
            # Find facts mentioning current entities.
            placeholders = ",".join("?" * len(current_entities))
            rows = conn.execute(
                f"""SELECT DISTINCT fact_id FROM fact_entity_mentions
                    WHERE entity_id IN ({placeholders}) AND tenant_id = ?""",
                (*current_entities, tenant_id),
            ).fetchall()
            new_facts = {r["fact_id"] for r in rows} - visited_facts
            visited_facts.update(new_facts)

            if hop + 1 < max_hops and new_facts:
                # Find related facts for the next hop.
                ph = ",".join("?" * len(new_facts))
                rel_rows = conn.execute(
                    f"""SELECT DISTINCT dst_fact_id AS fact_id FROM fact_relations
                        WHERE src_fact_id IN ({ph}) AND tenant_id = ?
                    UNION
                    SELECT DISTINCT src_fact_id AS fact_id FROM fact_relations
                        WHERE dst_fact_id IN ({ph}) AND tenant_id = ?""",
                    (*new_facts, tenant_id, *new_facts, tenant_id),
                ).fetchall()
                visited_facts.update(r["fact_id"] for r in rel_rows)

        return list(visited_facts)

    def related_facts_sync(
        self,
        fact_id: str,
        tenant_id: str,
    ) -> list[tuple[str, str, float]]:
        """Return (fact_id, relation, weight) for facts related to *fact_id*."""
        rows = self._db.query_all(
            """SELECT dst_fact_id AS fid, relation, confidence AS weight FROM fact_relations
               WHERE src_fact_id = ? AND tenant_id = ?
               UNION ALL
               SELECT src_fact_id AS fid, relation, confidence AS weight FROM fact_relations
               WHERE dst_fact_id = ? AND tenant_id = ?""",
            (fact_id, tenant_id, fact_id, tenant_id),
        )
        return [(r["fid"], r["relation"], r["weight"]) for r in rows]

    def facts_for_entity(
        self,
        entity_id: str,
        tenant_id: str = "default",
    ) -> list[str]:
        """Direct (hop-1) facts mentioning an entity."""
        rows = self._db.query_all(
            "SELECT DISTINCT fact_id FROM fact_entity_mentions WHERE entity_id = ? AND tenant_id = ?",
            (entity_id, tenant_id),
        )
        return [r["fact_id"] for r in rows]

    # ----- Async GraphBackend ABC -----

    async def upsert_entity_mention(
        self, fact_id: str, entity_id: str, role: str, scope: MemoryScope,
    ) -> None:
        tenant = scope.tenant_id or "default"
        self.upsert_entity_mention_sync(fact_id, entity_id, role, tenant)

    async def upsert_fact_relation(
        self, src_fact_id: str, dst_fact_id: str, relation: str,
        weight: float, scope: MemoryScope,
    ) -> None:
        tenant = scope.tenant_id or "default"
        self.upsert_fact_relation_sync(
            src_fact_id, dst_fact_id,
            relation,  # type: ignore[arg-type]
            tenant, weight,
        )

    async def entity_neighbors(
        self, entity_id: str, scope: MemoryScope, max_hops: int = 2,
    ) -> list[str]:
        tenant = scope.tenant_id or "default"
        return self.entity_neighbors_sync(entity_id, tenant, max_hops)

    async def related_facts(
        self, fact_id: str, scope: MemoryScope,
    ) -> list[tuple[str, str, float]]:
        tenant = scope.tenant_id or "default"
        return self.related_facts_sync(fact_id, tenant)


# ---------------------------------------------------------------------------
# GraphProjector — async edge projection from transaction outbox
# ---------------------------------------------------------------------------

class GraphProjector:
    """Project entity/relation edges from facts via the transaction outbox.

    The write pipeline enqueues outbox records when facts are inserted/updated;
    this projector polls the outbox and projects edges.  In Stage 3 this runs
    synchronously in tests; in production it runs as an async background task.
    """

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db

    def enqueue_projection(
        self,
        fact_id: str,
        tenant_id: str,
        op: str = "insert",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Enqueue a projection job in the transaction outbox."""
        outbox_id = f"ob_{uuid.uuid4().hex[:16]}"
        self._db.execute(
            """INSERT INTO transaction_outbox
               (id, tenant_id, fact_id, op, payload, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                outbox_id, tenant_id, fact_id, op,
                json.dumps(payload or {}, ensure_ascii=False),
                _utc_now_iso(),
            ),
        )

    def process_outbox(self, limit: int = 100) -> int:
        """Process pending outbox records. Returns count processed.

        For each record, re-projects entity edges from the fact's text.
        """
        from memory.entity_linker import EntityLinker
        from memory.backends.sqlite_backend import SQLiteStorageBackend

        linker = EntityLinker(self._db)
        backend = SQLiteStorageBackend(self._db)

        rows = self._db.query_all(
            """SELECT * FROM transaction_outbox
               WHERE processed_at IS NULL AND op != 'reindex'
               ORDER BY created_at LIMIT ?""",
            (limit,),
        )
        count = 0
        for row in rows:
            try:
                fact = backend.get_fact(row["fact_id"], row["tenant_id"])
                if fact is not None and row["op"] in ("insert", "update"):
                    linker.link_fact(fact)
                elif row["op"] == "delete":
                    self._db.execute(
                        "DELETE FROM fact_entity_mentions WHERE fact_id = ? AND tenant_id = ?",
                        (row["fact_id"], row["tenant_id"]),
                    )
                self._db.execute(
                    "UPDATE transaction_outbox SET processed_at = ? WHERE id = ?",
                    (_utc_now_iso(), row["id"]),
                )
                count += 1
            except Exception:
                # Mark as processed to avoid retry loop; log in production.
                self._db.execute(
                    "UPDATE transaction_outbox SET processed_at = ? WHERE id = ?",
                    (_utc_now_iso(), row["id"]),
                )
        return count

    def pending_count(self) -> int:
        row = self._db.query_one(
            "SELECT COUNT(*) AS n FROM transaction_outbox WHERE processed_at IS NULL"
        )
        return row["n"] if row else 0
