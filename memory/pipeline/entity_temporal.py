"""Stage 6: Entity & Temporal — Draft → mentions + TemporalInfo (§7.1).

Two sub-steps:
1. **Entity linking**: extract entity candidates from the fact text, resolve
   them to canonical entities (read-only — actual writes happen in the commit
   stage), and build ``EntityMention`` records on the draft.
2. **Temporal parsing**: parse temporal expressions from the fact text and
   set ``fact.temporal``.

This stage is *read-only* with respect to the database: it populates the
draft's ``entities`` and ``mentions`` lists, but the commit stage (stage 9)
writes them within the transaction boundary.  Entity resolution does reads
(alias lookup, fuzzy match) but creates no rows.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from memory.db import MemoryDB, memory_db
from memory.models import Entity, EntityMention, EntityType, Fact
from memory.pipeline import FactDraft


logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entity_id(name: str, tenant_id: str) -> str:
    """Deterministic entity_id from name + tenant (mirrors entity_linker)."""
    digest = hashlib.sha256(f"{tenant_id}:{name}".encode("utf-8")).hexdigest()
    return f"ent_{digest[:16]}"


def _bigrams(text: str) -> set[str]:
    text = text.strip().lower()
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


class EntityTemporalProcessor:
    """Stage 6: entity linking (read-only) + temporal parsing.

    Parameters:
        db: MemoryDB for read-only entity resolution.
        fuzzy_threshold: Jaccard bigram similarity threshold.
    """

    def __init__(
        self,
        db: MemoryDB | None = None,
        fuzzy_threshold: float = 0.45,
    ) -> None:
        self._db = db or memory_db
        self._fuzzy_threshold = fuzzy_threshold
        from memory.temporal_parser import TemporalParser
        self._temporal_parser = TemporalParser()

    def process(self, draft: FactDraft) -> None:
        """Populate ``draft.entities``, ``draft.mentions``, and ``fact.temporal``."""
        fact = draft.fact

        # --- Entity linking (read-only resolution) ---
        self._link_entities(draft)

        # --- Temporal parsing ---
        try:
            temporal = self._temporal_parser.parse(fact.text)
            if temporal is not None:
                fact.temporal = temporal
        except Exception:
            logger.exception("temporal parsing failed for fact %s", fact.id)

    # ----- Entity linking -----

    def _link_entities(self, draft: FactDraft) -> None:
        """Extract and resolve entities, populating draft.entities/mentions."""
        from memory.entity_linker import EntityLinker

        fact = draft.fact
        linker = EntityLinker(self._db, fuzzy_threshold=self._fuzzy_threshold)
        candidates = linker.extract_candidates(fact.text)
        if not candidates:
            return

        conn = self._db.connect()
        new_entities: list[Entity] = []
        mentions: list[EntityMention] = []

        for idx, (name, etype) in enumerate(candidates):
            entity_id = self._resolve_readonly(
                conn, name, etype, fact.tenant_id,
            )
            if entity_id is None:
                # New entity — compute ID and prepare for commit stage to write.
                entity_id = _entity_id(name, fact.tenant_id)
                now = datetime.now(timezone.utc)
                new_entities.append(Entity(
                    entity_id=entity_id,
                    tenant_id=fact.tenant_id,
                    canonical_name=name,
                    type=etype,
                    aliases=[name],
                    first_seen_at=now,
                    last_seen_at=now,
                ))

            role = "subject" if idx == 0 else "mentions"
            mentions.append(EntityMention(
                tenant_id=fact.tenant_id,
                fact_id=fact.id,
                entity_id=entity_id,
                mention_text=name,
                role=role,  # type: ignore[arg-type]
                linking_confidence=0.8,
            ))

        draft.entities = new_entities
        draft.mentions = mentions

    def _resolve_readonly(
        self,
        conn: Any,
        name: str,
        etype: EntityType,
        tenant_id: str,
    ) -> str | None:
        """Resolve a mention to an existing entity (read-only).

        Returns the ``entity_id`` if an existing entity is found (exact alias
        or fuzzy match), or ``None`` if a new entity should be created.
        """
        # 1. Exact alias match.
        row = conn.execute(
            "SELECT entity_id FROM entity_aliases WHERE alias = ? AND tenant_id = ?",
            (name, tenant_id),
        ).fetchone()
        if row:
            return row["entity_id"]

        # 2. Fuzzy match against existing entity names of the same type.
        existing = conn.execute(
            "SELECT entity_id, canonical_name FROM entities "
            "WHERE tenant_id = ? AND type = ?",
            (tenant_id, etype),
        ).fetchall()
        name_bigrams = _bigrams(name)
        best_id: str | None = None
        best_sim = 0.0
        for erow in existing:
            sim = _jaccard(name_bigrams, _bigrams(erow["canonical_name"]))
            if sim > best_sim:
                best_sim = sim
                best_id = erow["entity_id"]
        if best_id is not None and best_sim >= self._fuzzy_threshold:
            return best_id

        # 3. No match — caller should create a new entity.
        return None
