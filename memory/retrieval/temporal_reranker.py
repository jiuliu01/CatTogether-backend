"""Temporal reranker (§8.3).

Post-fusion reranking based on temporal validity:

- ``valid_to`` expired facts do not enter "current state" results.
- as-of queries judge validity at a specified time.
- ``progress`` and ``snapshot`` kinds decay with age.
- ``preference``, ``constraint``, ``convention`` use no uniform half-life
  (they are assumed to persist until explicitly superseded).
- ``confidence`` and ``importance`` adjustments are small-range only.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from memory.models import Fact, RetrievalResult


logger = logging.getLogger(__name__)


# Kind-specific temporal decay (half-life in days).
# progress/snapshot decay; stable kinds do not decay.
_HALF_LIFE_DAYS: dict[str, float | None] = {
    "progress": 30.0,       # ~1 month half-life
    "snapshot": 14.0,       # ~2 week half-life
    "project_fact": None,   # stable — no decay
    "preference": None,     # persists until superseded
    "constraint": None,     # persists until superseded
    "convention": None,     # persists until superseded
    "goal": None,           # persists until achieved/superseded
    "outcome": 90.0,        # ~3 month half-life (outcomes age)
    "relation": None,       # structural — no decay
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_expired(fact: Fact, as_of: datetime | None = None) -> bool:
    """Check if a fact's temporal validity has ended.

    A fact is expired if:
    - ``fact.expires_at`` is in the past, OR
    - ``fact.temporal.valid_to`` is in the past (relative to *as_of* or now).
    """
    ref = as_of or _utc_now()
    if fact.expires_at is not None and fact.expires_at < ref:
        return True
    if fact.temporal is not None and fact.temporal.valid_to is not None:
        if fact.temporal.valid_to < ref:
            return True
    return False


def _temporal_decay(fact: Fact, as_of: datetime | None = None) -> float:
    """Compute a multiplicative decay factor in (0, 1].

    For decaying kinds (progress, snapshot, outcome), the factor is
    ``0.5 ** (age_days / half_life)``.  For stable kinds, the factor is 1.0.
    """
    half_life = _HALF_LIFE_DAYS.get(fact.kind)
    if half_life is None or half_life <= 0:
        return 1.0

    ref = as_of or _utc_now()
    # Use event_time if available, else created_at.
    base_time = fact.created_at
    if fact.temporal is not None and fact.temporal.event_time is not None:
        base_time = fact.temporal.event_time
    age_days = (ref - base_time).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    return math.pow(0.5, age_days / half_life)


class TemporalReranker:
    """Post-fusion temporal reranking (§8.3).

    - Drops expired facts from "current state" results.
    - Applies kind-specific temporal decay to the fused score.
    - Small-range confidence/importance adjustments.
    """

    def __init__(self, *, drop_expired: bool = True) -> None:
        self._drop_expired = drop_expired

    def rerank(
        self,
        results: list[RetrievalResult],
        *,
        as_of: datetime | None = None,
        intent: str = "recall",
    ) -> list[RetrievalResult]:
        """Apply temporal reranking to fused results.

        Parameters
        ----------
        as_of
            Reference time for validity checks.  None = now.
        intent
            Retrieval intent.  For "audit" intent, expired facts are
            retained (they are historical evidence); for other intents,
            expired facts are dropped.

        Returns
        -------
        list[RetrievalResult]
            Reranked and possibly filtered results, sorted by adjusted
            score descending.
        """
        reranked: list[RetrievalResult] = []
        for result in results:
            fact = result.fact

            # Expired fact handling.
            if self._drop_expired and intent != "audit":
                if _is_expired(fact, as_of):
                    continue

            # Temporal decay.
            decay = _temporal_decay(fact, as_of)

            # Small-range confidence/importance boost (±10%).
            conf_boost = 1.0 + (fact.confidence - 0.5) * 0.1
            imp_boost = 1.0 + (fact.importance - 0.5) * 0.1

            adjusted_score = result.score * decay * conf_boost * imp_boost

            # Record the adjustment in the explanation.
            explanation = dict(result.explanation)
            explanation["temporal_decay"] = round(decay, 4)
            explanation["conf_boost"] = round(conf_boost, 4)
            explanation["imp_boost"] = round(imp_boost, 4)
            if _is_expired(fact, as_of):
                explanation["expired"] = True

            reranked.append(RetrievalResult(
                fact=fact,
                score=adjusted_score,
                signals=result.signals,
                explanation=explanation,
            ))

        # Re-sort by adjusted score.
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked
