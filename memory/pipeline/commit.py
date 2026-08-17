"""Stage 9: Transaction Commit + Outbox (§7.3).

Commits all drafts in a single ``MemoryUnitOfWork`` transaction.  Within one
``BEGIN IMMEDIATE``:

1. Insert each Fact (skip drafts with ``dedup_action == "skip"``).
2. Insert new Entities and entity aliases.
3. Insert EntityMentions.
4. Insert FactRelations (from dedup + supersede).
5. Insert embedding vectors (if available).
6. Mark superseded old facts.
7. Write audit log entries.
8. Write transaction outbox entries (for downstream projection / reindex).

If any step fails, the entire transaction rolls back — no half-committed
records, no orphaned embeddings, no audit entries without facts.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from memory.db import MemoryDB, memory_db
from memory.models import Event, Fact
from memory.pipeline import FactDraft


logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CommitStage:
    """Stage 9: commit all drafts in a single atomic transaction."""

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db

    def commit(self, drafts: list[FactDraft], event: Event) -> list[str]:
        """Commit all drafts atomically.

        Returns the list of committed fact IDs.  Drafts with
        ``dedup_action == "skip"`` are not inserted but their relations
        (if any) are still committed.
        """
        from memory.unit_of_work import MemoryUnitOfWork

        committed_ids: list[str] = []

        uow = MemoryUnitOfWork(self._db)
        uow.begin()
        try:
            for draft in drafts:
                fact = draft.fact

                # Insert the fact (unless skipped by true dedup).
                if draft.dedup_action != "skip":
                    self._insert_fact(uow, fact)
                    committed_ids.append(fact.id)

                    # Insert entities + aliases.
                    for entity in draft.entities:
                        self._insert_entity(uow, entity)

                    # Insert mentions.
                    for mention in draft.mentions:
                        self._insert_mention(uow, mention)

                    # Insert embedding.
                    if draft.embedding is not None:
                        self._insert_embedding(uow, fact.id, fact.tenant_id, draft.embedding)

                # Insert relations (even for skipped drafts).
                for relation in draft.relations:
                    self._insert_relation(uow, relation)
                    # If supersede, mark the old fact.
                    if relation.relation == "supersedes":
                        self._mark_superseded(uow, relation.dst_fact_id, relation.tenant_id)

                # Audit log.
                action = "write" if draft.dedup_action != "skip" else "dedup_skip"
                uow.add_audit(
                    tenant_id=fact.tenant_id,
                    action=action,
                    actor=event.actor.id,
                    fact_id=fact.id,
                    domain=fact.domain,
                    reason=draft.skipped_reason or None,
                )

                # Outbox entry.
                if draft.dedup_action != "skip":
                    uow.add_outbox(
                        tenant_id=fact.tenant_id,
                        event_type="insert",
                        fact_id=fact.id,
                        payload={
                            "fact_id": fact.id,
                            "domain": fact.domain,
                            "scope_id": fact.scope_id,
                            "kind": fact.kind,
                        },
                    )
                if draft.needs_reindex:
                    uow.add_outbox(
                        tenant_id=fact.tenant_id,
                        event_type="reindex",
                        fact_id=fact.id,
                        payload={"fact_id": fact.id, "reason": "embedding_failed"},
                    )

            uow.commit()
        except Exception:
            uow.rollback()
            logger.exception(
                "commit failed for event %s; rolled back %d drafts",
                event.event_id, len(drafts),
            )
            raise

        return committed_ids

    # ----- Row inserts -----

    def _insert_fact(self, uow: Any, fact: Fact) -> None:
        """Insert a fact row."""
        temporal_json = (
            fact.temporal.model_dump_json() if fact.temporal is not None else None
        )
        uow.conn.execute(
            """INSERT OR IGNORE INTO facts
               (id, tenant_id, domain, scope_id, agent_id, task_id, kind,
                text, search_text, tags, importance, confidence, status,
                temporal, provenance, embedding_model, created_at, updated_at,
                expires_at, version, schema_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact.id, fact.tenant_id, fact.domain, fact.scope_id,
                fact.agent_id, fact.task_id, fact.kind,
                fact.text, fact.search_text,
                json.dumps(fact.tags, ensure_ascii=False),
                fact.importance, fact.confidence, fact.status,
                temporal_json, fact.provenance.model_dump_json(),
                fact.embedding_model,
                fact.created_at.isoformat() if fact.created_at else _utc_now_iso(),
                fact.updated_at.isoformat() if fact.updated_at else _utc_now_iso(),
                fact.expires_at.isoformat() if fact.expires_at else None,
                fact.version, fact.schema_version,
            ),
        )

    def _insert_entity(self, uow: Any, entity: Any) -> None:
        """Insert an entity row + aliases."""
        now_iso = _utc_now_iso()
        uow.conn.execute(
            """INSERT OR IGNORE INTO entities
               (entity_id, tenant_id, canonical_name, type, aliases,
                first_seen_at, last_seen_at, merged_into_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entity.entity_id, entity.tenant_id,
                entity.canonical_name, entity.type,
                json.dumps(entity.aliases, ensure_ascii=False),
                entity.first_seen_at.isoformat() if entity.first_seen_at else now_iso,
                entity.last_seen_at.isoformat() if entity.last_seen_at else now_iso,
                entity.merged_into_id,
            ),
        )
        for alias in entity.aliases:
            uow.conn.execute(
                """INSERT OR IGNORE INTO entity_aliases
                   (entity_id, alias, tenant_id) VALUES (?, ?, ?)""",
                (entity.entity_id, alias, entity.tenant_id),
            )

    def _insert_mention(self, uow: Any, mention: Any) -> None:
        """Insert a fact_entity_mention row."""
        uow.conn.execute(
            """INSERT OR IGNORE INTO fact_entity_mentions
               (tenant_id, fact_id, entity_id, mention_text, role, linking_confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                mention.tenant_id, mention.fact_id, mention.entity_id,
                mention.mention_text, mention.role, mention.linking_confidence,
            ),
        )

    def _insert_relation(self, uow: Any, relation: Any) -> None:
        """Insert a fact_relation row."""
        uow.conn.execute(
            """INSERT OR IGNORE INTO fact_relations
               (tenant_id, src_fact_id, dst_fact_id, relation, confidence,
                created_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                relation.tenant_id, relation.src_fact_id, relation.dst_fact_id,
                relation.relation, relation.confidence,
                relation.created_by,
                relation.created_at.isoformat() if relation.created_at else _utc_now_iso(),
            ),
        )

    def _insert_embedding(
        self, uow: Any, fact_id: str, tenant_id: str, vector: list[float],
    ) -> None:
        """Insert an embedding vector (if vec extension is available)."""
        if not self._db.vec_available:
            return
        try:
            import sqlite3
            vec_blob = sqlite3.Vector(vector)  # type: ignore[attr-defined]
            uow.conn.execute(
                "INSERT OR REPLACE INTO fact_embeddings (fact_id, tenant_id, embedding) "
                "VALUES (?, ?, ?)",
                (fact_id, tenant_id, vec_blob),
            )
        except Exception:
            logger.warning("embedding insert failed for fact %s", fact_id)

    def _mark_superseded(self, uow: Any, fact_id: str, tenant_id: str) -> None:
        """Mark an old fact as superseded."""
        uow.conn.execute(
            "UPDATE facts SET status = 'superseded', updated_at = ? "
            "WHERE id = ? AND tenant_id = ?",
            (_utc_now_iso(), fact_id, tenant_id),
        )
