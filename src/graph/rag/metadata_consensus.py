"""Summarize strong metadata consensus across RAG results."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_DEFAULT_FIELDS = ("status", "rating", "sentiment", "answer", "stance", "outcome", "author", "source_type")


def _validate_min_support(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("min_support must be a positive integer")
    return value


def _fields(fields: Iterable[str] | None) -> tuple[str, ...]:
    if fields is None:
        return _DEFAULT_FIELDS
    values = tuple(fields)
    for field in values:
        if not isinstance(field, str) or not field.strip():
            raise ValueError("fields must contain non-empty strings")
    return tuple(field.strip() for field in values)


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _value(result: Any, key: str) -> Any:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING and metadata_value is not None:
            return metadata_value
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING and unit_value is not None:
            return unit_value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            return unit_metadata.get(key, _MISSING)
    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string(_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _source(result: Any) -> str:
    return _string(_value(result, "source_project")) or "unknown"


def summarize_metadata_consensus(
    results: Iterable[Any],
    *,
    fields: Iterable[str] | None = None,
    min_support: int = 2,
) -> list[dict[str, Any]]:
    """Return metadata values supported by at least min_support results."""
    support = _validate_min_support(min_support)
    field_names = _fields(fields)
    result_list = list(results)
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"counts": Counter(), "ids": set(), "sources": set()}
    )

    for index, result in enumerate(result_list):
        for field in field_names:
            display = _string(_value(result, field))
            if display is None:
                continue
            normalized = display.casefold()
            bucket = buckets[(field, normalized)]
            bucket["counts"][display] += 1
            bucket["ids"].add(_id(result, index))
            bucket["sources"].add(_source(result))

    rows = []
    total = max(len(result_list), 1)
    for (field, _normalized), bucket in buckets.items():
        support_count = sum(bucket["counts"].values())
        if support_count < support:
            continue
        display = sorted(bucket["counts"].items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))[0][0]
        rows.append(
            {
                "field": field,
                "value": display,
                "support_count": support_count,
                "support_ratio": round(support_count / total, 6),
                "representative_results": sorted(bucket["ids"])[:3],
                "source_projects": sorted(bucket["sources"]),
            }
        )
    rows.sort(key=lambda item: (-item["support_count"], item["field"], item["value"].casefold(), item["value"]))
    return rows
