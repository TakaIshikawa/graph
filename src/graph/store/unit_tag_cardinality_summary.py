"""Summarize per-unit tag cardinality."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from statistics import median
from typing import Any


def summarize_unit_tag_cardinality(units: Iterable[Any]) -> dict[str, Any]:
    counts = [len(_tags(unit)) for unit in units]
    bucket_counts = {"zero": 0, "one": 0, "two_to_five": 0, "six_to_ten": 0, "over_ten": 0}
    for count in counts:
        if count == 0:
            bucket_counts["zero"] += 1
        elif count == 1:
            bucket_counts["one"] += 1
        elif count <= 5:
            bucket_counts["two_to_five"] += 1
        elif count <= 10:
            bucket_counts["six_to_ten"] += 1
        else:
            bucket_counts["over_ten"] += 1
    return {
        "total_units": len(counts),
        "min": min(counts, default=0),
        "max": max(counts, default=0),
        "average": round(sum(counts) / len(counts), 2) if counts else 0,
        "median": median(counts) if counts else 0,
        "zero_tag_units": bucket_counts["zero"],
        "bucket_counts": bucket_counts,
    }


def _tags(unit: Any) -> list[str]:
    raw = _get(unit, "tags")
    if raw is None:
        raw = _metadata(unit).get("tags")
    values = raw.split(",") if isinstance(raw, str) else _flatten(raw)
    return [_text(value) for value in values if _text(value)]


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _flatten(child)]
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _flatten(child)]
    return [value]


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""
