"""Summarize unit word count distributions by source."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_CONTENT_KEYS = ("content", "text", "body")
_SOURCE_KEYS = ("source", "source_project", "source_id", "source_key")


def summarize_unit_word_count_distribution(units: Iterable[Any]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    total = 0
    for unit in units:
        total += 1
        groups[_source(unit)].append(_word_count(_content(unit)))

    rows = []
    for source in sorted(groups, key=sort_key):
        counts = groups[source]
        rows.append(
            {
                "source": source,
                "unit_count": len(counts),
                "empty_count": sum(1 for count in counts if count == 0),
                "short_count": sum(1 for count in counts if 1 <= count <= 100),
                "medium_count": sum(1 for count in counts if 101 <= count <= 500),
                "long_count": sum(1 for count in counts if count >= 501),
                "min_words": min(counts),
                "max_words": max(counts),
                "average_words": round(sum(counts) / len(counts), 2),
            }
        )
    return {"total_units": total, "rows": rows}


def _content(unit: Any) -> str:
    for key in _CONTENT_KEYS:
        value = get(unit, key)
        if isinstance(value, str):
            return value
    meta = metadata(unit)
    for key in _CONTENT_KEYS:
        value = meta.get(key)
        if isinstance(value, str):
            return value
    return ""


def _source(unit: Any) -> str:
    meta = metadata(unit)
    for key in _SOURCE_KEYS:
        value = field_value(get(unit, key)) or field_value(meta.get(key))
        if value:
            return value
    return "unknown"


def _word_count(content: str) -> int:
    return len(content.split())
