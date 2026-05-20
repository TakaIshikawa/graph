"""Analyze stale or undated source concentration in RAG results."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import coerce_now, parse_date, result_id, rounded_ratio, string, value

_DATE_KEYS = (
    "updated_at",
    "published_at",
    "publication_date",
    "created_at",
    "source_date",
    "date",
)


def analyze_source_freshness_risk(
    results: Iterable[Any],
    *,
    now: Any = None,
    stale_after_days: int = 180,
) -> dict[str, Any]:
    """Return source-level freshness risk summaries for retrieved results."""
    if not isinstance(stale_after_days, int) or isinstance(stale_after_days, bool) or stale_after_days < 1:
        raise ValueError("stale_after_days must be a positive integer")

    today = coerce_now(now)
    rows = []
    for index, result in enumerate(results):
        source = _source(result)
        best_date = _best_date(result)
        age_days = None if best_date is None else max((today - best_date).days, 0)
        rows.append(
            {
                "id": result_id(result, index),
                "source": source,
                "date": best_date,
                "age_days": age_days,
                "is_missing_date": best_date is None,
                "is_stale": age_days is not None and age_days > stale_after_days,
            }
        )

    total = len(rows)
    if total == 0:
        return {
            "total_results": 0,
            "stale_count": 0,
            "missing_date_count": 0,
            "stale_ratio": 0.0,
            "missing_date_ratio": 0.0,
            "sources": [],
            "warnings": ["no_results"],
        }

    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_rows[row["source"]].append(row)

    summaries = []
    for source, items in source_rows.items():
        stale_count = sum(1 for item in items if item["is_stale"])
        missing_count = sum(1 for item in items if item["is_missing_date"])
        dated = [item for item in items if item["date"] is not None]
        summaries.append(
            {
                "source": source,
                "total_results": len(items),
                "stale_count": stale_count,
                "missing_date_count": missing_count,
                "stale_ratio": rounded_ratio(stale_count, len(items)),
                "missing_date_ratio": rounded_ratio(missing_count, len(items)),
                "oldest_date": min((item["date"] for item in dated), default=None).isoformat() if dated else None,
                "newest_date": max((item["date"] for item in dated), default=None).isoformat() if dated else None,
                "stale_result_ids": sorted((item["id"] for item in items if item["is_stale"]), key=_sort_key),
                "missing_date_result_ids": sorted((item["id"] for item in items if item["is_missing_date"]), key=_sort_key),
            }
        )

    stale_count = sum(1 for row in rows if row["is_stale"])
    missing_count = sum(1 for row in rows if row["is_missing_date"])
    warnings = []
    if missing_count:
        warnings.append("missing_dates")
    if rounded_ratio(stale_count, total) >= 0.5:
        warnings.append("stale_concentration")
    stale_by_source = Counter(row["source"] for row in rows if row["is_stale"])
    if stale_count and stale_by_source:
        _, dominant_stale_count = sorted(stale_by_source.items(), key=lambda item: (-item[1], _sort_key(item[0])))[0]
        if dominant_stale_count / stale_count >= 0.75 and stale_count > 1:
            warnings.append("single_source_stale_dominance")

    summaries.sort(key=lambda item: (-item["stale_ratio"], -item["missing_date_ratio"], _sort_key(item["source"])))
    return {
        "total_results": total,
        "stale_count": stale_count,
        "missing_date_count": missing_count,
        "stale_ratio": rounded_ratio(stale_count, total),
        "missing_date_ratio": rounded_ratio(missing_count, total),
        "sources": summaries,
        "warnings": warnings,
    }


def _source(result: Any) -> str:
    for key in ("source", "source_project", "source_id", "publisher", "domain"):
        text = string(value(result, key))
        if text is not None:
            return text
    return "unknown"


def _best_date(result: Any) -> date | None:
    dates = []
    for key in _DATE_KEYS:
        parsed = parse_date(value(result, key))
        if parsed is not None:
            dates.append(parsed)
    return max(dates) if dates else None


def _sort_key(value_: object) -> tuple[str, str]:
    text = "" if value_ is None else str(value_)
    return (text.casefold(), text)
