"""Analyze freshness buckets by source for RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import coerce_now, result_date, source_id

_BUCKETS = ("recent", "current", "aging", "stale", "undated")


def _bucket(day: date | None, today: date) -> str:
    if day is None:
        return "undated"
    age_days = max((today - day).days, 0)
    if age_days <= 30:
        return "recent"
    if age_days <= 365:
        return "current"
    if age_days <= 1095:
        return "aging"
    return "stale"


def _empty_counts() -> dict[str, int]:
    return {bucket: 0 for bucket in _BUCKETS}


def analyze_source_recency_mix(
    results: Iterable[Any],
    *,
    now: Any = None,
) -> dict[str, Any]:
    """Return overall and per-source recency bucket counts."""
    try:
        rows = list(results or [])
    except TypeError:
        rows = []

    today = coerce_now(now)
    overall = Counter({bucket: 0 for bucket in _BUCKETS})
    per_source: dict[str, Counter[str]] = {}
    dated_values: list[date] = []

    for result in rows:
        source = source_id(result) or "unknown_source"
        day = result_date(result)
        bucket = _bucket(day, today)
        overall[bucket] += 1
        per_source.setdefault(source, Counter({name: 0 for name in _BUCKETS}))[bucket] += 1
        if day is not None:
            dated_values.append(day)

    return {
        "total_results": len(rows),
        "as_of": today.isoformat(),
        "overall": dict(overall),
        "sources": [
            {
                "source": source,
                "total": sum(counter.values()),
                "buckets": dict(counter),
            }
            for source, counter in sorted(per_source.items())
        ],
        "oldest_date": min(dated_values).isoformat() if dated_values else None,
        "newest_date": max(dated_values).isoformat() if dated_values else None,
    }
