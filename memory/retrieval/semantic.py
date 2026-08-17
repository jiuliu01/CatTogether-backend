"""Semantic signal — sqlite-vec KNN search (§8).

Uses the ``fact_embeddings`` vec0 virtual table for approximate nearest
neighbour search.  Hard filtering (tenant_id) is applied in the vec0
WHERE clause; domain/scope/status filtering is done post-hoc by joining
back to the ``facts`` table.

If sqlite-vec is not available (extension not loaded), returns an empty
list — the hybrid retriever will fall back to BM25 + Entity only.
"""
from __future__ import annotations

import logging
from typing import Any

from memory.backends.sqlite_backend import SQLiteStorageBackend
from memory.backends.sqlite_vec_backend import SqliteVecBackend
from memory.db import MemoryDB, memory_db
from memory.embedder import Embedder, get_embedder
from memory.models import Fact, RetrievalQuery


logger = logging.getLogger(__name__)


class SemanticRetriever:
    """sqlite-vec KNN semantic retrieval signal (§8).

    Generates a query embedding via the configured embedder, then runs
    a KNN search against ``fact_embeddings``.  Falls back gracefully
    when sqlite-vec is unavailable.
    """

    def __init__(
        self,
        db: MemoryDB | None = None,
        embedder: Embedder | None = None,
        vec_backend: SqliteVecBackend | None = None,
        backend: SQLiteStorageBackend | None = None,
    ) -> None:
        self._db = db or memory_db
        self._embedder = embedder or get_embedder()
        self._backend = backend or SQLiteStorageBackend(self._db)
        self._vec_backend = vec_backend or SqliteVecBackend(self._db)

    @property
    def available(self) -> bool:
        """True when vector search is enabled and sqlite-vec is loaded."""
        from config import settings
        return settings.memory_vector_enabled and self._db.vec_available

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        top_k: int | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[tuple[str, float, Fact]]:
        """Run semantic KNN search.

        Parameters
        ----------
        query_embedding
            Pre-computed query embedding (skip embedder call).  If None,
            the embedder is called on ``query.text``.

        Returns
        -------
        list of (fact_id, cosine_similarity, fact)
            Sorted by cosine similarity descending (best first).
            Empty if sqlite-vec is unavailable or embedding fails.
        """
        if not self.available:
            return []
        k = top_k or query.top_k

        try:
            embedding = query_embedding or self._embedder.embed(query.text)
        except Exception:
            logger.exception("semantic retriever: embedding generation failed")
            return []

        try:
            hits = self._vec_backend.search_sync(
                embedding,
                query.tenant_id,
                domain=query.domain,
                scope_id=query.scope_id,
                top_k=k,
            )
        except Exception:
            logger.exception("semantic retriever: vec KNN search failed")
            return []

        # Fetch full Fact objects and apply post-filters.
        results: list[tuple[str, float, Fact]] = []
        for fact_id, cosine_score in hits:
            fact = self._backend.get_fact(fact_id, query.tenant_id)
            if fact is None:
                continue
            # status filter (vec search doesn't filter on status).
            if fact.status != "active":
                continue
            # allowed_scopes post-filter.
            if query.allowed_scopes:
                allowed = any(
                    s.domain == fact.domain and s.scope_id == fact.scope_id
                    for s in query.allowed_scopes
                )
                if not allowed:
                    continue
            # expires_at post-filter.
            if fact.expires_at is not None:
                from datetime import datetime, timezone
                if fact.expires_at < datetime.now(timezone.utc):
                    continue
            results.append((fact_id, cosine_score, fact))
        return results
