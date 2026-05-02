"""Summarize RAG/search result dictionaries into deterministic facet counts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()


def _validate_max_values(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("max_values must be a non-negative integer")
    return value


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    label = " ".join(str(value).strip().split())
    return label or None


def _unit_value(unit: Any, key: str) -> Any:
    if isinstance(unit, Mapping):
        return unit.get(key, _MISSING)
    return getattr(unit, key, _MISSING)


def _result_value(result: Mapping[str, Any], key: str) -> Any:
    value = result.get(key, _MISSING)
    if value is not _MISSING and value is not None:
        return value

    unit = result.get("unit", _MISSING)
    if unit is _MISSING or unit is None:
        return value
    nested_value = _unit_value(unit, key)
    if nested_value is not _MISSING:
        return nested_value
    return value


def _metadata_path_value(metadata: Mapping[str, Any], path: str) -> Any:
    current: Any = metadata
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _facet_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return str(value)


def _count_value(
    counts: Counter[str],
    values_by_key: dict[str, Any],
    value: Any,
) -> None:
    key = _facet_key(value)
    counts[key] += 1
    values_by_key.setdefault(key, value)


def _metadata_values(value: Any) -> list[Any]:
    if isinstance(value, list | tuple | set | frozenset):
        values_by_key: dict[str, Any] = {}
        for item in value:
            values_by_key.setdefault(_facet_key(item), item)
        return [values_by_key[key] for key in sorted(values_by_key)]
    return [value]


def _facet_list(
    counts: Counter[str],
    values_by_key: dict[str, Any],
    *,
    max_values: int,
) -> list[dict[str, Any]]:
    items = [
        {"value": values_by_key[key], "key": key, "count": count}
        for key, count in counts.items()
    ]
    items.sort(key=lambda item: (-item["count"], item["key"]))
    return items[:max_values]


def build_result_facets(
    results: Iterable[Mapping[str, Any]],
    metadata_keys: Iterable[str] | None = None,
    max_values: int = 20,
) -> dict[str, Any]:
    """Return deterministic facet counts for RAG/search result payloads.

    Flat result fields take precedence over optional nested ``unit`` fields.
    Tags and multi-valued metadata entries are counted at most once per result.
    """
    max_values_value = _validate_max_values(max_values)
    metadata_key_list = [str(key) for key in (metadata_keys or [])]
    result_list = list(results)

    source_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    source_values: dict[str, Any] = {}
    content_values: dict[str, Any] = {}
    tag_values: dict[str, Any] = {}
    metadata_counts = {key: Counter() for key in metadata_key_list}
    metadata_values = {key: {} for key in metadata_key_list}

    for result in result_list:
        if not isinstance(result, Mapping):
            continue

        source_project = _string_value(_result_value(result, "source_project"))
        if source_project is not None:
            _count_value(source_counts, source_values, source_project)

        content_type = _string_value(_result_value(result, "content_type"))
        if content_type is not None:
            _count_value(content_counts, content_values, content_type)

        raw_tags = _result_value(result, "tags")
        if isinstance(raw_tags, Iterable) and not isinstance(raw_tags, str | bytes):
            tag_values_for_result = {
                tag
                for raw_tag in raw_tags
                if (tag := _string_value(raw_tag)) is not None
            }
            for tag in tag_values_for_result:
                _count_value(tag_counts, tag_values, tag)

        metadata = _result_value(result, "metadata")
        if not isinstance(metadata, Mapping):
            continue
        for key in metadata_key_list:
            value = _metadata_path_value(metadata, key)
            if value is _MISSING:
                continue
            value_counts = metadata_counts[key]
            values_by_key = metadata_values[key]
            for item in _metadata_values(value):
                _count_value(value_counts, values_by_key, item)

    return {
        "source_project": _facet_list(
            source_counts,
            source_values,
            max_values=max_values_value,
        ),
        "content_type": _facet_list(
            content_counts,
            content_values,
            max_values=max_values_value,
        ),
        "tags": _facet_list(tag_counts, tag_values, max_values=max_values_value),
        "metadata": {
            key: _facet_list(
                metadata_counts[key],
                metadata_values[key],
                max_values=max_values_value,
            )
            for key in metadata_key_list
        },
        "stats": {
            "result_count": len(result_list),
            "max_values": max_values_value,
            "metadata_keys": metadata_key_list,
        },
    }
