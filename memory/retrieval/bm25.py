"""BM25 signal — FTS5 search over ``search_text`` (§8).

Hard filtering (§8.1) is applied *before* candidate generation:
    tenant_id, allowed_scopes, status='active', expires_at.

The FTS5 virtual table ``facts_fts`` indexes the ``search_text`` column
(not ``text``).  Every Fact must have ``search_text`` populated or it will
not appear in BM25 results.

This module wraps the existing ``SQLiteStorageBackend.search_bm25`` which
already implements the FTS5 MATCH + tenant/domain/scope/status filtering.
"""
from __future__ import annotations

import logging

from memory.backends.sqlite_backend import SQLiteStorageBackend
from memory.db import MemoryDB, memory_db
from memory.models import Fact, RetrievalQuery


logger = logging.getLogger(__name__)


class BM25Retriever:
    """FTS5 BM25 retrieval signal (§8).

    Returns ranked (fact_id, score, Fact) triples.  Hard filtering
    (tenant, domain, scope, status=active) is applied in SQL.
    """

    def __init__(
        self,
        db: MemoryDB | None = None,
        backend: SQLiteStorageBackend | None = None,
    ) -> None:
        self._db = db or memory_db
        self._backend = backend or SQLiteStorageBackend(self._db)

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        top_k: int | None = None,
    ) -> list[tuple[str, float, Fact]]:
        """Run BM25 search.

        Returns
        -------
        list of (fact_id, bm25_score, fact)
            Sorted by BM25 relevance (best first).  Scores are the raw
            FTS5 bm25 values (negated so higher = more relevant).
        """
        if not query.text.strip():
            return []
        k = top_k or query.top_k

        hits = self._backend.search_bm25(
            query.text,
            query.tenant_id,
            domain=query.domain,
            scope_id=query.scope_id,
            top_k=k,
        )
        # search_bm25 returns (Fact, score) where score is the raw FTS5 value.
        # FTS5 bm25() is negative (more negative = better); negate for consistency.
        results: list[tuple[str, float, Fact]] = []
        for fact, raw_score in hits:
            # Apply allowed_scopes post-filter (defensive, §8.1).
            if query.allowed_scopes:
                allowed = any(
                    s.domain == fact.domain and s.scope_id == fact.scope_id
                    for s in query.allowed_scopes
                )
                if not allowed:
                    continue
            # Apply expires_at post-filter (defensive).
            if fact.expires_at is not None:
                from datetime import datetime, timezone
                if fact.expires_at < datetime.now(timezone.utc):
                    continue
            results.append((fact.id, -raw_score, fact))
        return results
