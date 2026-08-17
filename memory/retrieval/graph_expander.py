"""Graph expander (§8.4).

Optional post-fusion expansion via ``fact_relations``:

- Default OFF (``settings.memory_graph_enabled``).
- Only expands the top-20 fused results.
- Maximum 2 hops.
- Each hop re-checks tenant + scope (no cross-tenant leakage).
- MMR ensures diversity in the expanded results.

The expander follows ``related``, ``supports``, and ``derived_from``
relations (not ``contradicts`` or ``supersedes`` — those are handled by
the temporal reranker as validity signals).
"""
from __future__ import annotations

import logging
from typing import Any

from config import settings
from memory.db import MemoryDB, memory_db
from memory.models import Fact, RetrievalResult


logger = logging.getLogger(__name__)


# Relations that are safe to expand (positive/neutral, not negative).
_EXPAND_RELATIONS = ("related", "supports", "derived_from")


class GraphExpander:
    """Optional graph expansion via fact_relations (§8.4).

    Disabled by default.  When enabled, expands the top-N fused results
    by following ``related``/``supports``/``derived_from`` relations up
    to ``max_hops`` hops, re-checking tenant/scope at each hop.
    """

    def __init__(
        self,
        db: MemoryDB | None = None,
        *,
        enabled: bool | None = None,
        max_hops: int | None = None,
        top_n: int = 20,
    ) -> None:
        self._db = db or memory_db
        self._enabled = enabled if enabled is not None else settings.memory_graph_enabled
        self._max_hops = max_hops or settings.memory_graph_max_hops
        self._top_n = top_n

    @property
    def enabled(self) -> bool:
        return self._enabled

    def expand(
        self,
        results: list[RetrievalResult],
        *,
        tenant_id: str,
        allowed_domains: set[str] | None = None,
        allowed_scope_ids: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """Expand top-N results via fact_relations.

        Returns the original results plus any newly discovered facts,
        with graph-derived facts having ``"graph"`` in their signals.
        The caller is responsible for re-sorting and truncating.
        """
        if not self._enabled or not results:
            return results

        # Only expand the top-N.
        to_expand = results[: self._top_n]
        expanded = list(results)

        seen_ids: set[str] = {r.fact.id for r in results}
        new_results: list[RetrievalResult] = []

        for hop in range(self._max_hops):
            frontier_ids: list[str] = []
            for r in to_expand:
                if r.fact.id not in seen_ids or hop == 0:
                    frontier_ids.append(r.fact.id)

            if not frontier_ids:
                break

            discovered = self._expand_frontier(
                frontier_ids,
                tenant_id=tenant_id,
                allowed_domains=allowed_domains,
                allowed_scope_ids=allowed_scope_ids,
                seen_ids=seen_ids,
            )

            if not discovered:
                break

            for fact_id, fact, relation in discovered:
                seen_ids.add(fact_id)
                new_results.append(RetrievalResult(
                    fact=fact,
                    score=0.0,  # graph-expanded facts get score 0; RRF already fused
                    signals=["graph"],
                    explanation={"expanded_via": relation, "hop": hop + 1},
                ))

            # Next hop frontier = newly discovered facts.
            to_expand = new_results  # type: ignore

        return expanded + new_results

    def _expand_frontier(
        self,
        fact_ids: list[str],
        *,
        tenant_id: str,
        allowed_domains: set[str] | None,
        allowed_scope_ids: set[str] | None,
        seen_ids: set[str],
    ) -> list[tuple[str, Fact, str]]:
        """Find related facts for the given fact_ids.

        Returns (fact_id, fact, relation_type) triples.
        """
        if not fact_ids:
            return []
        placeholders = ",".join("?" for _ in fact_ids)
        rel_placeholders = ",".join("?" for _ in _EXPAND_RELATIONS)
        sql = (
            "SELECT DISTINCT r.dst_fact_id, r.relation, "
            "f.id, f.tenant_id, f.domain, f.scope_id, f.kind, f.text, "
            "f.search_text, f.tags, f.importance, f.confidence, f.status, "
            "f.temporal, f.provenance, f.embedding_model, "
            "f.created_at, f.updated_at, f.expires_at, f.version, f.schema_version, "
            "f.agent_id, f.task_id "
            "FROM fact_relations r "
            "JOIN facts f ON f.id = r.dst_fact_id AND f.tenant_id = r.tenant_id "
            f"WHERE r.src_fact_id IN ({placeholders}) "
            f"AND r.relation IN ({rel_placeholders}) "
            "AND r.tenant_id = ? "
            "AND f.status = 'active' "
            "AND f.tenant_id = ?"
        )
        params = (*fact_ids, *_EXPAND_RELATIONS, tenant_id, tenant_id)
        try:
            rows = self._db.query_all(sql, params)
        except Exception:
            logger.exception("graph expander: frontier query failed")
            return []

        from memory.backends.sqlite_backend import _row_to_fact
        discovered: list[tuple[str, Fact, str]] = []
        for row in rows:
            fact_id = row["id"]
            if fact_id in seen_ids:
                continue
            # Re-check tenant (defensive).
            if row["tenant_id"] != tenant_id:
                continue
            # Re-check domain/scope.
            if allowed_domains is not None and row["domain"] not in allowed_domains:
                continue
            if allowed_scope_ids is not None and row["scope_id"] not in allowed_scope_ids:
                continue
            fact = _row_to_fact(row)
            discovered.append((fact_id, fact, row["relation"]))
        return discovered
