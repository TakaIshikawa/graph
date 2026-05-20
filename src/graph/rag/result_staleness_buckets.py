"""Bucket RAG results by source timestamp age."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import coerce_now, result_date, result_id, rounded_ratio

_BUCKETS = ("fresh", "recent", "aging", "stale", "undated")


def bucket_results_by_staleness(results: Iterable[Any], *, now: Any = None) -> dict[str, Any]:
    """Group results into deterministic age buckets."""
    today = coerce_now(now)
    members = {bucket: [] for bucket in _BUCKETS}
    dates: list[date] = []

    for index, result in enumerate(results):
        parsed = result_date(result)
        bucket = _bucket(parsed, today)
        result_id_ = result_id(result, index)
        members[bucket].append(result_id_)
        if parsed is not None:
            dates.append(parsed)

    total = sum(len(ids) for ids in members.values())
    counts = {bucket: len(members[bucket]) for bucket in _BUCKETS}
    ratios = {bucket: rounded_ratio(counts[bucket], total) for bucket in _BUCKETS}
    warnings = []
    if total == 0:
        warnings.append("no_results")
    if total and counts["undated"] / total >= 0.5:
        warnings.append("undated_results_dominate")
    if total and counts["stale"] / total >= 0.5:
        warnings.append("stale_results_dominate")

    return {
        "total_results": total,
        "bucket_counts": counts,
        "bucket_ratios": ratios,
        "result_ids": {bucket: sorted(members[bucket], key=_sort_key) for bucket in _BUCKETS},
        "oldest_date": min(dates).isoformat() if dates else None,
        "newest_date": max(dates).isoformat() if dates else None,
        "warnings": warnings,
    }


def _bucket(parsed: date | None, today: date) -> str:
    if parsed is None:
        return "undated"
    age = max((today - parsed).days, 0)
    if age <= 30:
        return "fresh"
    if age <= 90:
        return "recent"
    if age <= 180:
        return "aging"
    return "stale"


def _sort_key(value_: object) -> tuple[str, str]:
    text = "" if value_ is None else str(value_)
    return (text.casefold(), text)
