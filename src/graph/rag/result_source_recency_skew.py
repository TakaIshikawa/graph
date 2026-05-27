"""Analyze per-source recency skew across retrieved RAG results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import median
from typing import Any

from graph.rag._analysis_utils import coerce_now, domain_for, result_date, rounded_ratio, source_id, string, value


def analyze_result_source_recency_skew(results: Iterable[Any], reference_date: Any = None) -> dict[str, Any]:
    """Summarize result ages by source and flag materially stale source groups."""
    today = coerce_now(reference_date)
    groups: dict[str, list[int]] = defaultdict(list)
    dates: dict[str, list[Any]] = defaultdict(list)
    total_results = 0
    undated_count = 0

    for result in results or []:
        total_results += 1
        source = _source(result)
        dated = result_date(result)
        if dated is None:
            undated_count += 1
            continue
        groups[source].append(max((today - dated).days, 0))
        dates[source].append(dated)

    all_ages = [age for ages in groups.values() for age in ages]
    overall_median = float(median(all_ages)) if all_ages else None
    rows = []
    for source, ages in sorted(groups.items()):
        source_median = float(median(ages))
        rows.append(
            {
                "source": source,
                "result_count": len(ages),
                "newest_date": max(dates[source]).isoformat(),
                "oldest_date": min(dates[source]).isoformat(),
                "median_age_days": source_median,
                "stale_share": rounded_ratio(sum(1 for age in ages if age > (overall_median or 0)), len(ages)),
            }
        )

    skewed = [
        row
        for row in rows
        if overall_median is not None
        and row["result_count"] > 0
        and row["median_age_days"] >= overall_median + 30
        and row["stale_share"] >= 0.5
    ]
    skewed.sort(key=lambda row: (-row["median_age_days"], row["source"]))

    return {
        "total_results": total_results,
        "dated_count": len(all_ages),
        "undated_count": undated_count,
        "overall_median_age_days": overall_median,
        "source_summaries": rows,
        "skewed_sources": skewed,
    }


def _source(result: Any) -> str:
    for key in ("source", "source_id", "source_project", "provider", "domain", "source_domain"):
        text = string(value(result, key))
        if text:
            return text.casefold()
    return domain_for(result) or "unknown"
