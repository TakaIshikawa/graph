"""Summarize unit content lengths into deterministic buckets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_BUCKETS = ("empty", "short", "medium", "long", "very_long")


def summarize_unit_content_length_buckets(units: Iterable[Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter({bucket: 0 for bucket in _BUCKETS})
    for unit in units:
        content = "" if get(unit, "content") is None else str(get(unit, "content"))
        bucket = _bucket(len(content))
        counts[bucket] += 1
        rows.append({"unit_id": unit_id(unit), "content_length": len(content), "bucket": bucket})
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {"total_units": len(rows), "bucket_counts": dict(counts), "units": rows}


def _bucket(length: int) -> str:
    if length == 0:
        return "empty"
    if length <= 280:
        return "short"
    if length <= 2_000:
        return "medium"
    if length <= 10_000:
        return "long"
    return "very_long"
