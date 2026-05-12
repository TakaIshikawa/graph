"""Analyze entity coverage across RAG result sets."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_DEFAULT_ENTITY_KEYS = ("entity", "entities", "people", "authors", "organizations", "projects", "tags")
_VALUE_KEYS = ("name", "title", "label", "value", "entity", "author", "tag")


def _validate_limit(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("limit must be a non-negative integer")
    return value


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


def _normalize(value: str) -> str:
    return value.casefold()


def _iter_strings(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        for key in _VALUE_KEYS:
            string = _string(value.get(key, _MISSING))
            if string is not None:
                return [string]
        return [_string(key) for key in value if _string(key) is not None]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        values: set[str] = set()
        for item in value:
            values.update(_iter_strings(item))
        return sorted(values)
    string = _string(value)
    return [] if string is None else [string]


def _id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string(_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _source(result: Any) -> str:
    return _string(_value(result, "source_project")) or "unknown"


def _keys(entity_keys: Iterable[str] | None) -> tuple[str, ...]:
    if entity_keys is None:
        return _DEFAULT_ENTITY_KEYS
    keys = tuple(entity_keys)
    for key in keys:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("entity_keys must contain non-empty strings")
    return tuple(key.strip() for key in keys)


def analyze_result_entity_coverage(
    results: Iterable[Any],
    *,
    entity_keys: Iterable[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return stable ranked entity coverage rows for RAG results."""
    limit_value = _validate_limit(limit)
    key_names = _keys(entity_keys)
    result_list = list(results)
    counts: Counter[str] = Counter()
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    result_ids: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)

    for index, result in enumerate(result_list):
        row_entities: set[str] = set()
        display_by_key: dict[str, str] = {}
        for key in key_names:
            for entity in _iter_strings(_value(result, key)):
                normalized = _normalize(entity)
                row_entities.add(normalized)
                display_by_key.setdefault(normalized, entity)
        for normalized in row_entities:
            counts[normalized] += 1
            labels[normalized][display_by_key[normalized]] += 1
            result_ids[normalized].add(_id(result, index))
            sources[normalized].add(_source(result))

    rows = []
    total = max(len(result_list), 1)
    for normalized, count in counts.items():
        display = sorted(labels[normalized].items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))[0][0]
        rows.append(
            {
                "entity": display,
                "count": count,
                "result_ids": sorted(result_ids[normalized]),
                "source_projects": sorted(sources[normalized]),
                "coverage_ratio": round(count / total, 6),
            }
        )
    rows.sort(key=lambda item: (-item["count"], item["entity"].casefold(), item["entity"]))
    return rows[:limit_value]
