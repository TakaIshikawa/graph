"""Analyze evidence age buckets across RAG results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import coerce_now, result_date, rounded_ratio

_BUCKETS = ("fresh", "recent", "aging", "stale", "undated")


def analyze_result_evidence_age_mix(results: Iterable[Any], reference_date: Any = None) -> dict[str, Any]:
    """Bucket result evidence by age relative to a reference date."""
    today = coerce_now(reference_date)
    dates = []
    counts: Counter[str] = Counter()
    for result in list(results or []):
        dated = result_date(result)
        if dated is None:
            counts["undated"] += 1
            continue
        dates.append(dated)
        age_days = max((today - dated).days, 0)
        counts[_bucket(age_days)] += 1
    total = sum(counts.values())
    warnings = []
    if total and rounded_ratio(counts["undated"], total) >= 0.5:
        warnings.append("undated_heavy_evidence_mix")
    dated_total = total - counts["undated"]
    if dated_total and rounded_ratio(counts["stale"], dated_total) >= 0.5:
        warnings.append("stale_heavy_evidence_mix")
    return {
        "total_results": total,
        "counts": {bucket: counts[bucket] for bucket in _BUCKETS},
        "ratios": {bucket: rounded_ratio(counts[bucket], total) for bucket in _BUCKETS},
        "oldest_date": min(dates).isoformat() if dates else None,
        "newest_date": max(dates).isoformat() if dates else None,
        "warnings": warnings,
    }


def _bucket(age_days: int) -> str:
    if age_days <= 30:
        return "fresh"
    if age_days <= 180:
        return "recent"
    if age_days <= 365:
        return "aging"
    return "stale"
