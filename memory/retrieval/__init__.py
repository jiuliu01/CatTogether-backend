"""Three-signal hybrid retrieval pipeline (§8).

Signals:
    BM25 (FTS5) → Semantic (sqlite-vec KNN) → Entity (canonical mention)

Fusion: Weighted RRF (§8.2).  Post-fusion: Temporal rerank (§8.3),
optional Graph expansion + MMR (§8.4).

Usage::

    from memory.retrieval import HybridRetriever
    from memory.models import RetrievalQuery

    results = HybridRetriever().retrieve(query)
"""
from __future__ import annotations

from memory.retrieval.hybrid import HybridRetriever
from memory.retrieval.rrf import reciprocal_rank_fusion

__all__ = ["HybridRetriever", "reciprocal_rank_fusion"]
