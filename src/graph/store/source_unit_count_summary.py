"""Summarize unit counts by source identifier."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_SOURCE_KEYS = ("source_project", "source", "source_name", "source_type")
_MISSING_SOURCE = "missing"


def summarize_source_unit_counts(units: Iterable[Any], *, top_limit: int = 5) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total = 0
    missing = 0
    for unit in units:
        total += 1
        source = _source(unit)
        if source == _MISSING_SOURCE:
            missing += 1
        counts[source] += 1

    rows = [
        {"source": source, "unit_count": count}
        for source, count in sorted(counts.items(), key=lambda item: (-item[1], sort_key(item[0])))
    ]
    return {
        "total_units": total,
        "source_count": len([source for source in counts if source != _MISSING_SOURCE]),
        "missing_source_units": missing,
        "source_counts": {row["source"]: row["unit_count"] for row in rows},
        "top_sources": rows[:top_limit],
        "rows": rows,
    }


def _source(unit: Any) -> str:
    meta = metadata(unit)
    for key in _SOURCE_KEYS:
        value = field_value(get(unit, key))
        if value:
            return value
        meta_value = meta.get(key)
        if isinstance(meta_value, Mapping):
            continue
        value = field_value(meta_value)
        if value:
            return value
    source = meta.get("source")
    if isinstance(source, Mapping):
        for key in ("name", "type", "id"):
            value = field_value(source.get(key))
            if value:
                return value
    return _MISSING_SOURCE
