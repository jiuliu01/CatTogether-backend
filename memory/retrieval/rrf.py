"""Weighted Reciprocal Rank Fusion (§8.2).

``score[fact_id] += weight[signal] / (rrf_k + rank)``

Each signal contributes a rank (1-based) for each fact it returns.  The
fused score is the weighted sum of ``1 / (k + rank)`` across all signals
that returned the fact.  Higher score = more relevant.
"""
from __future__ import annotations

from typing import Sequence


def reciprocal_rank_fusion(
    signal_results: dict[str, Sequence[str]],
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
) -> dict[str, float]:
    """Fuse multiple ranked lists via Weighted RRF.

    Parameters
    ----------
    signal_results
        Mapping ``signal_name → ranked list of fact_ids`` (rank 0 = best).
        Each list is already sorted by the signal's internal score.
    weights
        Mapping ``signal_name → weight``.  Signals not in *weights* get
        weight 1.0.  Default weights: semantic=1.0, bm25=1.0, entity=0.8.
    rrf_k
        RRF constant (default 60).  Higher k smooths differences between
        ranks; lower k amplifies top results.

    Returns
    -------
    dict[str, float]
        Mapping ``fact_id → fused_score``, sorted descending by score.
    """
    if weights is None:
        weights = {"semantic": 1.0, "bm25": 1.0, "entity": 0.8}

    scores: dict[str, float] = {}
    for signal_name, ranked_ids in signal_results.items():
        w = weights.get(signal_name, 1.0)
        if w == 0.0:
            continue
        for rank, fact_id in enumerate(ranked_ids):
            # rank is 0-based; RRF uses 1-based rank.
            scores[fact_id] = scores.get(fact_id, 0.0) + w / (rrf_k + rank + 1)

    # Sort descending by score.
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
