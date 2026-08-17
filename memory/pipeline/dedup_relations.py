"""Stage 8: Dedup & Relations — Draft + existing Facts → DedupDecision (§7.2).

ADD-only write semantics (§7.2, R4):

- **True dedup** (skip): same source ID + content hash → the fact already
  exists from the same source; skip the insert.
- **Similar** (candidate): Jaccard/embedding similarity above threshold →
  find the candidate but *never overwrite*.  The new fact is still inserted;
  a ``related`` relation is created.
- **Contradicts**: the new fact contradicts an existing fact → insert the new
  fact and create a ``contradicts`` relation.
- **Supersedes**: the new fact is a trusted correction of an existing fact →
  insert the new fact, create a ``supersedes`` relation, and mark the old
  fact as ``superseded``.

Similarity never overwrites; it only finds candidates and builds relations.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

from memory.db import MemoryDB, memory_db
from memory.models import Fact, FactRelation
from memory.pipeline import DedupDecision, FactDraft, _content_hash


logger = logging.getLogger(__name__)


# Thresholds.
JACCARD_THRESHOLD = 0.65
"""Bigram Jaccard similarity above which two facts are considered "similar"."""
EMBEDDING_THRESHOLD = 0.85
"""Cosine similarity above which two facts are considered near-duplicates."""


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


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class DedupRelations:
    """Stage 8: evaluate dedup and build relations.

    Parameters:
        db: MemoryDB for querying existing facts.
        jaccard_threshold: text similarity threshold.
        embedding_threshold: vector similarity threshold.
    """

    def __init__(
        self,
        db: MemoryDB | None = None,
        jaccard_threshold: float = JACCARD_THRESHOLD,
        embedding_threshold: float = EMBEDDING_THRESHOLD,
    ) -> None:
        self._db = db or memory_db
        self._jaccard_threshold = jaccard_threshold
        self._embedding_threshold = embedding_threshold

    def fetch_candidates(self, draft: FactDraft) -> list[Fact]:
        """Fetch existing facts in the same scope for dedup comparison.

        Only active and pending_review facts in the same tenant/domain/scope
        are considered.
        """
        fact = draft.fact
        rows = self._db.query_all(
            """SELECT * FROM facts
               WHERE tenant_id = ? AND domain = ? AND scope_id = ?
                 AND status IN ('active', 'pending_review')
               ORDER BY created_at DESC LIMIT 50""",
            (fact.tenant_id, fact.domain, fact.scope_id),
        )
        return [self._row_to_fact(r) for r in rows]

    def evaluate(
        self,
        draft: FactDraft,
        existing: list[Fact],
    ) -> DedupDecision:
        """Evaluate the draft against existing facts.

        Returns a ``DedupDecision`` with:
        - ``action``: "add" (insert), "skip" (true dedup), or "relation_only"
        - ``relations``: fact relations to create
        """
        fact = draft.fact
        relations: list[FactRelation] = []

        # Compute the draft's content hash.
        draft_hash = _content_hash(
            fact.text, fact.kind, fact.domain, fact.scope_id,
        )
        draft_bigrams = _bigrams(fact.text)

        for old_fact in existing:
            # --- True dedup: same source + content hash → skip ---
            old_hash = _content_hash(
                old_fact.text, old_fact.kind,
                old_fact.domain, old_fact.scope_id,
            )
            if old_hash == draft_hash:
                # Check if same source IDs.
                draft_sources = set(fact.provenance.source_ids)
                old_sources = set(old_fact.provenance.source_ids)
                if draft_sources and draft_sources == old_sources:
                    return DedupDecision(
                        action="skip",
                        reason="true dedup: same source + content hash",
                    )

            # --- Similarity: find candidates, never overwrite ---
            text_sim = _jaccard(draft_bigrams, _bigrams(old_fact.text))
            emb_sim = 0.0
            if draft.embedding is not None:
                old_emb = self._fetch_embedding(old_fact.id, old_fact.tenant_id)
                if old_emb is not None:
                    emb_sim = _cosine(draft.embedding, old_emb)

            is_similar = (
                text_sim >= self._jaccard_threshold
                or emb_sim >= self._embedding_threshold
            )

            if is_similar:
                # Check for contradiction (negation in new text).
                if self._is_contradiction(fact.text, old_fact.text):
                    relations.append(FactRelation(
                        tenant_id=fact.tenant_id,
                        src_fact_id=fact.id,
                        dst_fact_id=old_fact.id,
                        relation="contradicts",
                        confidence=max(text_sim, emb_sim),
                        created_by=fact.provenance.author or "system",
                    ))
                # Check for supersession (trusted correction).
                elif self._is_correction(fact, old_fact):
                    relations.append(FactRelation(
                        tenant_id=fact.tenant_id,
                        src_fact_id=fact.id,
                        dst_fact_id=old_fact.id,
                        relation="supersedes",
                        confidence=max(text_sim, emb_sim),
                        created_by=fact.provenance.author or "system",
                    ))
                else:
                    # Just related — both facts coexist.
                    relations.append(FactRelation(
                        tenant_id=fact.tenant_id,
                        src_fact_id=fact.id,
                        dst_fact_id=old_fact.id,
                        relation="related",
                        confidence=max(text_sim, emb_sim),
                        created_by=fact.provenance.author or "system",
                    ))

        if relations and not any(r.relation == "supersedes" for r in relations):
            # Similar facts found but we still insert the new one.
            return DedupDecision(action="add", relations=relations)

        return DedupDecision(action="add", relations=relations)

    # ----- Helpers -----

    def _fetch_embedding(self, fact_id: str, tenant_id: str) -> list[float] | None:
        """Fetch an existing fact's embedding vector."""
        if not self._db.vec_available:
            return None
        try:
            row = self._db.query_one(
                "SELECT embedding FROM fact_embeddings WHERE fact_id = ? AND tenant_id = ?",
                (fact_id, tenant_id),
            )
            if row is None:
                return None
            import sqlite3
            blob = row["embedding"]
            if isinstance(blob, (bytes, bytearray)):
                import struct
                n = len(blob) // 4
                return list(struct.unpack(f"{n}f", blob))
            return None
        except Exception:
            return None

    def _is_contradiction(self, new_text: str, old_text: str) -> bool:
        """Heuristic: check if new_text negates old_text.

        Looks for negation markers (不, 没有, 并非, not, never, don't) in the
        new text that aren't in the old text.
        """
        negations = ("不", "没有", "并非", "不是", "不对", " not ", " never ", " don't ")
        new_has_neg = any(neg in new_text.lower() for neg in negations)
        old_has_neg = any(neg in old_text.lower() for neg in negations)
        return new_has_neg and not old_has_neg

    def _is_correction(self, new_fact: Fact, old_fact: Fact) -> bool:
        """Check if new_fact is a trusted correction of old_fact.

        A correction is a new fact from a trusted source (user/system) that
        has higher confidence than the old fact.
        """
        trusted_sources = {"user_statement", "manual", "migration"}
        new_trusted = new_fact.provenance.source_type in trusted_sources
        higher_conf = new_fact.confidence > old_fact.confidence
        return new_trusted and higher_conf

    def _row_to_fact(self, row: Any) -> Fact:
        """Reconstruct a Fact from a database row."""
        import json
        from datetime import datetime
        from memory.models import ProvenanceInfo, TemporalInfo

        temporal = None
        if row["temporal"]:
            temporal = TemporalInfo.model_validate_json(row["temporal"])

        provenance = ProvenanceInfo.model_validate_json(row["provenance"])

        return Fact(
            id=row["id"],
            tenant_id=row["tenant_id"],
            domain=row["domain"],  # type: ignore[arg-type]
            scope_id=row["scope_id"],
            agent_id=row["agent_id"],
            task_id=row["task_id"],
            kind=row["kind"],  # type: ignore[arg-type]
            text=row["text"],
            search_text=row["search_text"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            importance=row["importance"],
            confidence=row["confidence"],
            status=row["status"],  # type: ignore[arg-type]
            temporal=temporal,
            provenance=provenance,
            embedding_model=row["embedding_model"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            version=row["version"],
            schema_version=row["schema_version"],
        )
