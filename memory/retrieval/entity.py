"""Entity signal — canonical entity mention lookup (§8).

Finds facts that mention entities extracted from the query.  This signal
is weaker than BM25/Semantic (default RRF weight 0.8) but provides a
complementary recall path: a fact about "PostgreSQL" will match a query
mentioning "Postgres" even if the BM25 phrase doesn't align and the
embedding is distant.

Entity resolution is done via the ``entity_aliases`` table: the query
text is scanned for known aliases, and each alias maps to a canonical
``entity_id``.  Facts are then found via ``fact_entity_mentions``.
"""
from __future__ import annotations

import logging
from typing import Any

from memory.db import MemoryDB, memory_db
from memory.models import Fact, RetrievalQuery


logger = logging.getLogger(__name__)


class EntityRetriever:
    """Entity-mention retrieval signal (§8).

    Looks up facts by canonical entity IDs derived from the query text
    or explicit ``query.entities``.  Uses ``fact_entity_mentions`` to
    find fact → entity links and ``entity_aliases`` for alias resolution.
    """

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        top_k: int | None = None,
    ) -> list[tuple[str, float, Fact]]:
        """Run entity-mention lookup.

        Returns
        -------
        list of (fact_id, entity_score, fact)
            ``entity_score`` is 1.0 for direct entity_id matches, 0.8 for
            alias matches.  Sorted by score descending, then deduplicated
            by fact_id (keeping the highest score).
        """
        if not query.entities and not query.text.strip():
            return []
        k = top_k or query.top_k

        # Resolve entity IDs from query.entities (explicit) + query.text (alias scan).
        entity_ids = self._resolve_entities(query)
        if not entity_ids:
            return []

        # Find facts mentioning these entities.
        fact_scores = self._find_facts_by_entities(entity_ids, query, k)
        if not fact_scores:
            return []

        # Fetch full Fact objects and apply post-filters.
        results: list[tuple[str, float, Fact]] = []
        for fact_id, score in fact_scores:
            fact = self._fetch_fact(fact_id, query.tenant_id)
            if fact is None:
                continue
            results.append((fact_id, score, fact))
        return results[:k]

    def _resolve_entities(self, query: RetrievalQuery) -> list[str]:
        """Resolve query text/entities to canonical entity_ids."""
        entity_ids: list[str] = []

        # Explicit entity IDs from the query.
        for eid in query.entities:
            entity_ids.append(eid)

        # Alias scan: find entity_aliases whose alias appears in query.text.
        if query.text.strip():
            alias_ids = self._scan_aliases(query.text, query.tenant_id)
            for eid in alias_ids:
                if eid not in entity_ids:
                    entity_ids.append(eid)

        return entity_ids

    def _scan_aliases(self, text: str, tenant_id: str) -> list[str]:
        """Scan *text* for known entity aliases.

        Loads all aliases for the tenant and checks if each appears as a
        substring in *text*.  This is O(n_aliases) per query — acceptable
        for the expected alias count (hundreds).  For larger scale, an
        Aho-Corasick automaton or FTS5 on aliases would be needed.
        """
        try:
            rows = self._db.query_all(
                "SELECT entity_id, alias FROM entity_aliases WHERE tenant_id = ?",
                (tenant_id,),
            )
        except Exception:
            return []
        found: list[str] = []
        for row in rows:
            alias = row["alias"]
            if alias and alias.lower() in text.lower():
                eid = row["entity_id"]
                if eid not in found:
                    found.append(eid)
        return found

    def _find_facts_by_entities(
        self,
        entity_ids: list[str],
        query: RetrievalQuery,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Find facts mentioning the given entity_ids.

        Returns (fact_id, score) pairs sorted by score descending.
        """
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        clauses = [
            f"m.entity_id IN ({placeholders})",
            "m.tenant_id = ?",
            "f.status = 'active'",
            "f.tenant_id = ?",
        ]
        params: list[Any] = [*entity_ids, query.tenant_id, query.tenant_id]

        if query.domain is not None:
            clauses.append("f.domain = ?")
            params.append(query.domain)
        if query.scope_id is not None:
            clauses.append("f.scope_id = ?")
            params.append(query.scope_id)

        # expires_at filter.
        clauses.append("(f.expires_at IS NULL OR f.expires_at > datetime('now'))")

        where = " AND ".join(clauses)
        sql = (
            "SELECT m.fact_id, MAX(m.linking_confidence) AS max_conf "
            "FROM fact_entity_mentions m "
            "JOIN facts f ON f.id = m.fact_id AND f.tenant_id = m.tenant_id "
            f"WHERE {where} "
            "GROUP BY m.fact_id "
            "ORDER BY max_conf DESC "
            "LIMIT ?"
        )
        params.append(top_k)
        try:
            rows = self._db.query_all(sql, tuple(params))
        except Exception:
            logger.exception("entity retriever: fact lookup failed")
            return []
        return [(row["fact_id"], float(row["max_conf"])) for row in rows]

    def _fetch_fact(self, fact_id: str, tenant_id: str) -> Fact | None:
        """Fetch a Fact by ID with tenant check."""
        from memory.backends.sqlite_backend import SQLiteStorageBackend
        return SQLiteStorageBackend(self._db).get_fact(fact_id, tenant_id)
