"""Temporal parsing for Memory 2.0 (§6 环5, Stage 3).

Extracts temporal metadata from fact text and populates the ``TemporalInfo``
embedded field.  Supports:

- ISO 8601 dates: 2026-08-10, 2026-08
- Chinese date expressions: 2026年8月10日, 8月10日, 上周, 昨天, 今天
- Relative time: yesterday, today, tomorrow, last week, next month

The parser is rule-based (regex + dateutil) for determinism.  LLM-based
extraction is a future enhancement.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from memory.models import TemporalInfo


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_ISO_YEAR_MONTH = re.compile(r"\b(\d{4})-(\d{1,2})\b")
_CN_DATE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_CN_MONTH_DAY = re.compile(r"(\d{1,2})月(\d{1,2})日")

# Chinese relative expressions → (delta_days, label)
_CN_RELATIVE: list[tuple[re.Pattern[str], timedelta, str]] = [
    (re.compile(r"今天|今日"), timedelta(days=0), "today"),
    (re.compile(r"昨天|昨日"), timedelta(days=-1), "yesterday"),
    (re.compile(r"明天|明日"), timedelta(days=1), "tomorrow"),
    (re.compile(r"前天"), timedelta(days=-2), "day_before_yesterday"),
    (re.compile(r"后天"), timedelta(days=2), "day_after_tomorrow"),
    (re.compile(r"上周|上个星期"), timedelta(weeks=-1), "last_week"),
    (re.compile(r"下周|下个星期"), timedelta(weeks=1), "next_week"),
    (re.compile(r"上个月"), timedelta(days=-30), "last_month"),
    (re.compile(r"下个月"), timedelta(days=30), "next_month"),
]

# English relative expressions
_EN_RELATIVE: list[tuple[re.Pattern[str], timedelta, str]] = [
    (re.compile(r"\btoday\b", re.IGNORECASE), timedelta(days=0), "today"),
    (re.compile(r"\byesterday\b", re.IGNORECASE), timedelta(days=-1), "yesterday"),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), timedelta(days=1), "tomorrow"),
    (re.compile(r"\blast\s+week\b", re.IGNORECASE), timedelta(weeks=-1), "last_week"),
    (re.compile(r"\bnext\s+week\b", re.IGNORECASE), timedelta(weeks=1), "next_week"),
    (re.compile(r"\blast\s+month\b", re.IGNORECASE), timedelta(days=-30), "last_month"),
    (re.compile(r"\bnext\s+month\b", re.IGNORECASE), timedelta(days=30), "next_month"),
]


class TemporalParser:
    """Parse temporal expressions from text into TemporalInfo (§6 环5)."""

    def parse(self, text: str, *, now: datetime | None = None) -> TemporalInfo | None:
        """Extract temporal metadata from *text*.

        Returns TemporalInfo if any temporal expression is found, else None.
        """
        now = now or _utc_now()
        expressions: list[str] = []
        event_time: datetime | None = None
        valid_from: datetime | None = None
        valid_to: datetime | None = None

        # --- ISO dates ---
        for m in _ISO_DATE.finditer(text):
            try:
                dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
            except ValueError:
                continue
            if event_time is None:
                event_time = dt
            expressions.append(m.group(0))

        # --- ISO year-month ---
        for m in _ISO_YEAR_MONTH.finditer(text):
            if m.group(0) in [e for e in expressions]:
                continue
            try:
                dt = datetime(int(m.group(1)), int(m.group(2)), 1, tzinfo=timezone.utc)
            except ValueError:
                continue
            if event_time is None:
                event_time = dt
            expressions.append(m.group(0))

        # --- Chinese dates ---
        for m in _CN_DATE.finditer(text):
            try:
                dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
            except ValueError:
                continue
            if event_time is None:
                event_time = dt
            expressions.append(m.group(0))

        for m in _CN_MONTH_DAY.finditer(text):
            try:
                dt = datetime(now.year, int(m.group(1)), int(m.group(2)), tzinfo=timezone.utc)
            except ValueError:
                continue
            if event_time is None:
                event_time = dt
            expressions.append(m.group(0))

        # --- Chinese relative ---
        for pattern, delta, label in _CN_RELATIVE:
            if pattern.search(text):
                dt = now + delta
                if event_time is None:
                    event_time = dt
                expressions.append(label)

        # --- English relative ---
        for pattern, delta, label in _EN_RELATIVE:
            if pattern.search(text):
                dt = now + delta
                if event_time is None:
                    event_time = dt
                expressions.append(label)

        if not expressions and event_time is None:
            return None

        return TemporalInfo(
            valid_from=valid_from,
            valid_to=valid_to,
            event_time=event_time,
            time_expressions=expressions,
            is_snapshot=False,
        )
