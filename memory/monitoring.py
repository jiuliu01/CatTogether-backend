"""Monitoring metrics for Memory 2.0 (§13, Stage 4).

Lightweight, dependency-free metrics collection.  All metrics are aggregated
in-process and exposed via ``snapshot()`` for scraping or logging.

Security constraint (§13): metrics MUST NOT leak memory text content.
Only counts, latencies, and categorical labels (domain, kind, reason) are
recorded — never fact text, tags, or entity names.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

@dataclass
class Counter:
    """A simple monotonic counter."""
    _value: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._value += n

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


@dataclass
class Histogram:
    """A latency histogram with bucketed counts.

    Buckets are in milliseconds: [0, 1, 5, 10, 50, 100, 500, 1000, 5000, +inf].
    """
    _counts: list[int] = field(default_factory=lambda: [0] * 10)
    _sum_ms: float = 0.0
    _n: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    _BUCKETS_MS: tuple[float, ...] = (1, 5, 10, 50, 100, 500, 1000, 5000)

    def observe_ms(self, ms: float) -> None:
        with self._lock:
            self._sum_ms += ms
            self._n += 1
            for i, bound in enumerate(self._BUCKETS_MS):
                if ms <= bound:
                    self._counts[i] += 1
                    return
            self._counts[len(self._BUCKETS_MS)] += 1  # overflow bucket

    @property
    def count(self) -> int:
        with self._lock:
            return self._n

    @property
    def mean_ms(self) -> float:
        with self._lock:
            return self._sum_ms / self._n if self._n > 0 else 0.0

    def buckets(self) -> dict[str, int]:
        with self._lock:
            labels = ["<=1ms", "<=5ms", "<=10ms", "<=50ms", "<=100ms",
                      "<=500ms", "<=1s", "<=5s", ">5s"]
            return {label: count for label, count in zip(labels, self._counts)}


class LabelledCounter:
    """A counter keyed by a label tuple (e.g. (domain, kind, reason))."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, ...], int] = defaultdict(int)
        self._lock = threading.Lock()

    def inc(self, *labels: str, n: int = 1) -> None:
        key = tuple(labels)
        with self._lock:
            self._counts[key] += n

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"|".join(k): v for k, v in self._counts.items()}

    @property
    def total(self) -> int:
        with self._lock:
            return sum(self._counts.values())


# ---------------------------------------------------------------------------
# MemoryMetrics registry
# ---------------------------------------------------------------------------

class MemoryMetrics:
    """Central metrics registry for the memory system.

    Thread-safe singleton.  Access via ``memory_metrics``.
    """

    def __init__(self) -> None:
        # Operation counts
        self.write_total = Counter()
        self.search_total = Counter()
        self.update_total = Counter()
        self.delete_total = Counter()
        self.forget_total = Counter()

        # Write outcomes (domain, kind, reason) → count
        self.write_outcomes = LabelledCounter()
        # Search signal hits (signal_name) → count
        self.search_signals = LabelledCounter()
        # Dedup / discard reasons → count
        self.discard_reasons = LabelledCounter()

        # Latency histograms
        self.write_latency = Histogram()
        self.search_latency = Histogram()
        self.update_latency = Histogram()
        self.embedding_latency = Histogram()
        self.retrieval_latency = Histogram()

        # Hot memory
        self.hot_promoted = Counter()
        self.hot_approved = Counter()
        self.hot_archived = Counter()
        self.hot_evicted = Counter()

        # Skill
        self.skill_created = Counter()
        self.skill_updated = Counter()
        self.skill_rolled_back = Counter()

        # Graph projection
        self.graph_projected = Counter()
        self.graph_pending = Counter()

        # Permission denials (role, capability) → count
        self.permission_denials = LabelledCounter()

    def record_write(
        self,
        domain: str,
        kind: str,
        accepted: bool,
        reason: str = "",
        latency_ms: float = 0.0,
    ) -> None:
        """Record a write operation outcome.  No text content is stored."""
        self.write_total.inc()
        outcome = "accepted" if accepted else "rejected"
        self.write_outcomes.inc(domain, kind, outcome, reason or "none")
        if latency_ms > 0:
            self.write_latency.observe_ms(latency_ms)

    def record_search(
        self,
        domain: str | None,
        result_count: int,
        signals: list[str],
        latency_ms: float = 0.0,
    ) -> None:
        """Record a search operation.  No query text is stored."""
        self.search_total.inc()
        for signal in signals:
            self.search_signals.inc(signal)
        # result_count as a coarse label
        self.search_signals.inc("result_count_bucket", _count_bucket(result_count))
        if latency_ms > 0:
            self.search_latency.observe_ms(latency_ms)

    def record_discard(self, reason: str) -> None:
        """Record a pipeline discard (dedup, conflict, validation)."""
        self.discard_reasons.inc(reason)

    def record_permission_denial(self, role: str, capability: str) -> None:
        """Record a permission denial."""
        self.permission_denials.inc(role, capability)

    def snapshot(self) -> dict[str, Any]:
        """Return a flat dict of all metric values for logging/scraping."""
        return {
            "write_total": self.write_total.value,
            "search_total": self.search_total.value,
            "update_total": self.update_total.value,
            "delete_total": self.delete_total.value,
            "forget_total": self.forget_total.value,
            "write_outcomes": self.write_outcomes.snapshot(),
            "search_signals": self.search_signals.snapshot(),
            "discard_reasons": self.discard_reasons.snapshot(),
            "write_latency_ms": {
                "count": self.write_latency.count,
                "mean": round(self.write_latency.mean_ms, 3),
                "buckets": self.write_latency.buckets(),
            },
            "search_latency_ms": {
                "count": self.search_latency.count,
                "mean": round(self.search_latency.mean_ms, 3),
                "buckets": self.search_latency.buckets(),
            },
            "embedding_latency_ms": {
                "count": self.embedding_latency.count,
                "mean": round(self.embedding_latency.mean_ms, 3),
            },
            "retrieval_latency_ms": {
                "count": self.retrieval_latency.count,
                "mean": round(self.retrieval_latency.mean_ms, 3),
            },
            "hot_promoted": self.hot_promoted.value,
            "hot_approved": self.hot_approved.value,
            "hot_archived": self.hot_archived.value,
            "hot_evicted": self.hot_evicted.value,
            "skill_created": self.skill_created.value,
            "skill_updated": self.skill_updated.value,
            "skill_rolled_back": self.skill_rolled_back.value,
            "graph_projected": self.graph_projected.value,
            "graph_pending": self.graph_pending.value,
            "permission_denials": self.permission_denials.snapshot(),
        }

    def reset(self) -> None:
        """Reset all metrics (for tests)."""
        for attr in dir(self):
            obj = getattr(self, attr, None)
            if isinstance(obj, Counter):
                obj._value = 0
            elif isinstance(obj, Histogram):
                obj._counts = [0] * 10
                obj._sum_ms = 0.0
                obj._n = 0
            elif isinstance(obj, LabelledCounter):
                obj._counts.clear()


def _count_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 3:
        return "1-3"
    if n <= 10:
        return "4-10"
    return "11+"


# Module-level singleton
memory_metrics = MemoryMetrics()


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

class Timer:
    """Context manager that measures elapsed time in milliseconds."""

    def __init__(self) -> None:
        self.ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0
