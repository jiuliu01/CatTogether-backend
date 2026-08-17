"""Fact layer store — long-term factual knowledge (§5.2).

Provides CRUD + search over the ``facts`` table.  Search is delegated to
the three-signal hybrid retriever (§8).  This is the main retrieval layer
in the context builder.
"""
from __future__ import annotations

from typing import Any

from memory.backends.sqlite_backend import SQLiteStorageBackend
from memory.db import MemoryDB, memory_db
from memory.models import Fact, RetrievalQuery, RetrievalResult


class FactStore:
    """Fact layer (§5.2).

    Wraps ``SQLiteStorageBackend`` for CRUD and the hybrid retriever for
    search.  The context builder calls ``search()`` with the main budget.
    """

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db
        self._backend = SQLiteStorageBackend(self._db)

    def get(self, fact_id: str, tenant_id: str = "default") -> Fact | None:
        """Get a single fact by ID (with tenant check)."""
        return self._backend.get_fact(fact_id, tenant_id)

    def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Three-signal hybrid search (§8).

        Delegates to ``HybridRetriever.retrieve()``.
        """
        from memory.retrieval import HybridRetriever
        return HybridRetriever(self._db).retrieve(query)

    def list_by_scope(
        self,
        tenant_id: str,
        domain: str,
        scope_id: str,
        *,
        status: str = "active",
        limit: int = 100,
    ) -> list[Fact]:
        """List facts by scope (no relevance ranking, just listing)."""
        rows = self._db.query_all(
            """SELECT * FROM facts
               WHERE tenant_id = ? AND domain = ? AND scope_id = ? AND status = ?
               ORDER BY updated_at DESC LIMIT ?""",
            (tenant_id, domain, scope_id, status, limit),
        )
        from memory.backends.sqlite_backend import _row_to_fact
        return [_row_to_fact(row) for row in rows]

    def update_metadata(
        self,
        fact_id: str,
        tenant_id: str,
        metadata: dict[str, Any],
    ) -> Fact | None:
        """Update only the metadata/tags of a fact (not the text)."""
        fact = self.get(fact_id, tenant_id)
        if fact is None:
            return None
        # Only tags and importance/confidence can be patched via metadata.
        update_fields: dict[str, Any] = {}
        if "tags" in metadata:
            import json as _json
            update_fields["tags"] = _json.dumps(metadata["tags"])
        if "importance" in metadata:
            update_fields["importance"] = float(metadata["importance"])
        if "confidence" in metadata:
            update_fields["confidence"] = float(metadata["confidence"])
        if not update_fields:
            return fact
        # Build SET clause.
        sets = ", ".join(f"{k} = ?" for k in update_fields)
        params = list(update_fields.values())
        params.append(fact_id)
        params.append(tenant_id)
        self._db.execute(
            f"UPDATE facts SET {sets}, updated_at = datetime('now') "
            "WHERE id = ? AND tenant_id = ?",
            tuple(params),
        )
        return self.get(fact_id, tenant_id)
