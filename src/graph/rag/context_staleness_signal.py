"""Analyze date freshness signals in packed RAG context items."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import MISSING, parse_date, result_id, value


def analyze_context_staleness_signals(context_items: Iterable[Any], reference_date: str | date | None = None, stale_after_days: int = 365) -> dict[str, Any]:
    rows = list(context_items or [])
    ref = parse_date(reference_date) if reference_date is not None else date.today()
    if ref is None:
        ref = date.today()
    counts: Counter[str] = Counter({"recent": 0, "stale": 0, "missing_date": 0, "invalid_date": 0, "future_date": 0})
    stale_items: list[str] = []
    missing: list[str] = []
    future: list[str] = []
    invalid: list[str] = []
    for index, item in enumerate(rows):
        rid = result_id(item, index)
        raw = _date_value(item)
        if raw is None:
            counts["missing_date"] += 1
            missing.append(rid)
            continue
        parsed = parse_date(raw)
        if parsed is None:
            counts["invalid_date"] += 1
            invalid.append(rid)
            continue
        age = (ref - parsed).days
        if age < 0:
            counts["future_date"] += 1
            future.append(rid)
        elif age > stale_after_days:
            counts["stale"] += 1
            stale_items.append(rid)
        else:
            counts["recent"] += 1
    return {"total_items": len(rows), "freshness_buckets": dict(counts), "stale_item_ids": stale_items, "missing_date_item_ids": missing, "invalid_date_item_ids": invalid, "future_date_item_ids": future}


def _date_value(item: Any) -> Any:
    for key in ("updated_at", "published_at", "publication_date", "date", "timestamp"):
        current = value(item, key)
        if current is not MISSING and current is not None:
            return current
    return None
