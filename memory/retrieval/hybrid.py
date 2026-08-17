"""Hybrid retrieval orchestrator (§8).

Orchestrates the three-signal retrieval pipeline:

    Query + Actor
          │
          ▼
    PermissionService.compile_allowed_scopes()
          │
          ├─ Semantic KNN ─┐
          ├─ BM25          ├─> Weighted RRF
          └─ Entity        ┘
                               │
                               ▼
                 Temporal / Validity / Quality Rerank
                               │
                               ▼
                 Optional Graph Expansion + MMR
                               │
                               ▼
                            Top-K Facts

Hard filtering (§8.1): every signal applies tenant + allowed_scopes +
status=active + expires_at filtering *before* candidate generation, then
a defensive post-filter is applied after fusion.
"""
from __future__ import annotations

import logging
from typing import Any

from config import settings
from memory.db import MemoryDB, memory_db
from memory.embedder import Embedder, get_embedder
from memory.models import RetrievalQuery, RetrievalResult
from memory.permissions import compile_allowed_scopes
from memory.retrieval.bm25 import BM25Retriever
from memory.retrieval.entity import EntityRetriever
from memory.retrieval.graph_expander import GraphExpander
from memory.retrieval.mmr import mmr_rerank
from memory.retrieval.rrf import reciprocal_rank_fusion
from memory.retrieval.semantic import SemanticRetriever
from memory.retrieval.temporal_reranker import TemporalReranker


logger = logging.getLogger(__name__)


class HybridRetriever:
    """Three-signal hybrid retrieval with RRF fusion (§8).

    Signals: BM25 (FTS5) + Semantic (sqlite-vec KNN) + Entity (mention).
    Fusion: Weighted RRF (k=60, weights 1.0/1.0/0.8).
    Post-fusion: Temporal rerank → optional Graph expansion → MMR.
    """

    def __init__(
        self,
        db: MemoryDB | None = None,
        embedder: Embedder | None = None,
        *,
        bm25: BM25Retriever | None = None,
        semantic: SemanticRetriever | None = None,
        entity: EntityRetriever | None = None,
        temporal_reranker: TemporalReranker | None = None,
        graph_expander: GraphExpander | None = None,
    ) -> None:
        self._db = db or memory_db
        self._embedder = embedder or get_embedder()
        self._bm25 = bm25 or BM25Retriever(self._db)
        self._semantic = semantic or SemanticRetriever(self._db, self._embedder)
        self._entity = entity or EntityRetriever(self._db)
        self._temporal = temporal_reranker or TemporalReranker()
        self._graph = graph_expander or GraphExpander(self._db)

    @property
    def db(self) -> MemoryDB:
        return self._db

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def vec_available(self) -> bool:
        return self._db.vec_available

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        query_embedding: list[float] | None = None,
        apply_mmr: bool = True,
        mmr_lambda: float = 0.7,
    ) -> list[RetrievalResult]:
        """Run the full hybrid retrieval pipeline.

        Parameters
        ----------
        query_embedding
            Pre-computed query embedding (skip embedder call for semantic).
        apply_mmr
            Whether to apply MMR diversity reranking.  Default True.
        mmr_lambda
            MMR relevance/diversity trade-off.  Default 0.7.

        Returns
        -------
        list[RetrievalResult]
            Top-K results sorted by fused+reranked score descending.
        """
        # Compile allowed scopes if not already provided.
        if not query.allowed_scopes and query.actor is not None:
            query = query.model_copy(update={
                "allowed_scopes": [
                    s for s in compile_allowed_scopes(
                        query.actor, query.tenant_id,
                        domain=query.domain, scope_id=query.scope_id,
                    )
                ],
            })

        # --- Stage 1: Run three signals ---
        bm25_hits = self._bm25.retrieve(query)
        sem_hits = self._semantic.retrieve(query, query_embedding=query_embedding)
        entity_hits = self._entity.retrieve(query)

        # --- Stage 2: Weighted RRF fusion ---
        # Build ranked ID lists for each signal.
        bm25_ranked = [fid for fid, _, _ in bm25_hits]
        sem_ranked = [fid for fid, _, _ in sem_hits]
        entity_ranked = [fid for fid, _, _ in entity_hits]

        signal_results: dict[str, list[str]] = {}
        if bm25_ranked:
            signal_results["bm25"] = bm25_ranked
        if sem_ranked:
            signal_results["semantic"] = sem_ranked
        if entity_ranked:
            signal_results["entity"] = entity_ranked

        fused_scores = reciprocal_rank_fusion(
            signal_results,
            weights={
                "semantic": settings.memory_rrf_weight_semantic,
                "bm25": settings.memory_rrf_weight_bm25,
                "entity": settings.memory_rrf_weight_entity,
            },
            rrf_k=settings.memory_rrf_k,
        )

        if not fused_scores:
            return []

        # --- Stage 3: Build RetrievalResults ---
        # Collect all facts from all signals.
        fact_map: dict[str, Any] = {}  # fact_id → Fact
        signal_map: dict[str, list[str]] = {}  # fact_id → signals
        for fid, _, fact in bm25_hits:
            fact_map[fid] = fact
            signal_map.setdefault(fid, []).append("bm25")
        for fid, _, fact in sem_hits:
            fact_map[fid] = fact
            signal_map.setdefault(fid, []).append("semantic")
        for fid, _, fact in entity_hits:
            fact_map[fid] = fact
            signal_map.setdefault(fid, []).append("entity")

        results: list[RetrievalResult] = []
        for fid, score in fused_scores.items():
            fact = fact_map.get(fid)
            if fact is None:
                continue
            # Defensive post-filter (§8.1): re-check tenant + status.
            if fact.tenant_id != query.tenant_id:
                continue
            if fact.status != "active":
                continue
            # Deduplicate signals.
            signals = list(dict.fromkeys(signal_map.get(fid, [])))
            results.append(RetrievalResult(
                fact=fact,
                score=score,
                signals=signals,
                explanation={"rrf_score": round(score, 6)},
            ))

        # --- Stage 4: Temporal rerank ---
        results = self._temporal.rerank(
            results,
            as_of=query.as_of,
            intent=query.intent,
        )

        # --- Stage 5: Optional Graph expansion ---
        if self._graph.enabled:
            allowed_domains = {s.domain for s in query.allowed_scopes} if query.allowed_scopes else None
            allowed_scope_ids = {s.scope_id for s in query.allowed_scopes} if query.allowed_scopes else None
            results = self._graph.expand(
                results,
                tenant_id=query.tenant_id,
                allowed_domains=allowed_domains,
                allowed_scope_ids=allowed_scope_ids,
            )
            # Re-sort after expansion (graph facts have score 0).
            results.sort(key=lambda r: r.score, reverse=True)

        # --- Stage 6: MMR diversity ---
        if apply_mmr and len(results) > 1:
            results = mmr_rerank(
                results,
                lambda_param=mmr_lambda,
                top_k=query.top_k,
            )

        return results[: query.top_k]

    def retrieve_with_embeddings(
        self,
        query: RetrievalQuery,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve with a pre-computed query embedding."""
        return self.retrieve(query, query_embedding=query_embedding)
