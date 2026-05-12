"""Build source-project overlap rows from shared RAG metadata values."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_DEFAULT_KEYS = ("tags", "entities", "topics", "authors", "urls")
_VALUE_KEYS = ("name", "title", "label", "value", "url", "tag", "entity", "topic", "author")


def _validate_min_overlap(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("min_overlap must be a non-negative integer")
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


def _source(result: Any) -> str:
    return _string(_value(result, "source_project")) or "unknown"


def _keys(keys: Iterable[str] | None) -> tuple[str, ...]:
    if keys is None:
        return _DEFAULT_KEYS
    values = tuple(keys)
    for key in values:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("keys must contain non-empty strings")
    return tuple(key.strip() for key in values)


def _normalized_values(result: Any, keys: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    for key in keys:
        for value in _iter_strings(_value(result, key)):
            values.add(value.casefold())
    return values


def build_source_overlap_matrix(
    results: Iterable[Any],
    *,
    keys: Iterable[str] | None = None,
    min_overlap: int = 1,
) -> list[dict[str, Any]]:
    """Compare source_project groups by shared normalized metadata values."""
    key_names = _keys(keys)
    min_overlap_value = _validate_min_overlap(min_overlap)
    source_values: dict[str, set[str]] = defaultdict(set)

    for result in results:
        source_values[_source(result)].update(_normalized_values(result, key_names))

    rows = []
    sources = sorted(source_values)
    for left_index, source_a in enumerate(sources):
        for source_b in sources[left_index + 1 :]:
            left = source_values[source_a]
            right = source_values[source_b]
            shared = sorted(left & right)
            if len(shared) < min_overlap_value:
                continue
            union_count = len(left | right)
            rows.append(
                {
                    "source_a": source_a,
                    "source_b": source_b,
                    "overlap_count": len(shared),
                    "shared_values": shared,
                    "source_a_count": len(left),
                    "source_b_count": len(right),
                    "jaccard": round(len(shared) / union_count, 6) if union_count else 0,
                }
            )
    rows.sort(key=lambda item: (-item["overlap_count"], item["source_a"], item["source_b"]))
    return rows
