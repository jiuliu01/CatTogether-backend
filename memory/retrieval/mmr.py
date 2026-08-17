"""MMR (Maximal Marginal Relevance) diversity reranker.

Ensures result diversity by penalising redundancy with already-selected
results.  Used after graph expansion (§8.4) to avoid a cluster of
near-identical facts dominating the top-K.

``score_mmr = λ * relevance - (1-λ) * max_sim_to_selected``
"""
from __future__ import annotations

import math
from typing import Sequence


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors.  Returns 0.0 for empty/mismatched."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def _text_jaccard(a: str, b: str) -> float:
    """Jaccard similarity between token sets of two strings."""
    if not a or not b:
        return 0.0
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def mmr_rerank(
    results: list,  # list[RetrievalResult]
    *,
    lambda_param: float = 0.7,
    top_k: int | None = None,
    embeddings: dict[str, list[float]] | None = None,
) -> list:
    """Rerank results using MMR for diversity.

    Parameters
    ----------
    results
        List of RetrievalResult, sorted by relevance descending.
    lambda_param
        Trade-off between relevance (1.0 = pure relevance, no diversity)
        and diversity (0.0 = pure diversity).  Default 0.7.
    top_k
        Number of results to return.  Default = len(results).
    embeddings
        Optional mapping ``fact_id → embedding vector`` for cosine-based
        similarity.  If None, falls back to Jaccard text similarity.

    Returns
    -------
    list[RetrievalResult]
        Reranked results selected via MMR.
    """
    if not results:
        return []
    k = top_k or len(results)
    if len(results) <= 1:
        return list(results)

    # Normalise relevance scores to [0, 1].
    max_score = max(r.score for r in results) or 1.0
    relevance = {r.fact.id: r.score / max_score for r in results}

    selected: list = []
    selected_ids: set[str] = set()
    remaining = list(results)

    # Always pick the top-relevance result first.
    remaining.sort(key=lambda r: r.score, reverse=True)
    first = remaining.pop(0)
    selected.append(first)
    selected_ids.add(first.fact.id)

    while remaining and len(selected) < k:
        best_idx = -1
        best_mmr = -float("inf")

        for i, candidate in enumerate(remaining):
            # Max similarity to already-selected results.
            max_sim = 0.0
            for sel in selected:
                if embeddings is not None:
                    emb_c = embeddings.get(candidate.fact.id)
                    emb_s = embeddings.get(sel.fact.id)
                    if emb_c and emb_s:
                        sim = _cosine_similarity(emb_c, emb_s)
                    else:
                        sim = _text_jaccard(candidate.fact.text, sel.fact.text)
                else:
                    sim = _text_jaccard(candidate.fact.text, sel.fact.text)
                if sim > max_sim:
                    max_sim = sim

            mmr_score = lambda_param * relevance[candidate.fact.id] - (1 - lambda_param) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx < 0:
            break
        selected.append(remaining.pop(best_idx))

    return selected
