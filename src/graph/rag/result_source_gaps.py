"""Detect missing or weak source attribution in RAG results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_ID_KEYS = ("id", "result_id", "unit_id", "source_id")
_FIELD_KEYS = {
    "source": ("source", "source_name", "source_project", "publisher", "domain"),
    "url": ("url", "source_url", "canonical_url", "external_url", "link", "permalink", "uri"),
    "title": ("title", "name", "headline"),
    "author": ("author", "authors", "creator", "byline", "owner"),
}
_UNKNOWN_SOURCE_VALUES = {"unknown", "none", "null", "n/a", "na", "unspecified", "untitled"}


def detect_result_source_gaps(results: Iterable[Any]) -> dict[str, Any]:
    """Return aggregate and per-result source attribution gaps."""
    result_list = list(results)
    counts = {
        "missing_source": 0,
        "missing_url": 0,
        "missing_title": 0,
        "missing_author": 0,
        "complete_attribution": 0,
    }
    rows = []

    for index, result in enumerate(result_list):
        values = {field: _field_text(result, field) for field in _FIELD_KEYS}
        missing = []
        for field, value in values.items():
            if not _has_field_value(field, value):
                missing.append(field)
                counts[f"missing_{field}"] += 1
        if not missing:
            counts["complete_attribution"] += 1
        rows.append(
            {
                "result_id": _result_id(result, index),
                "title": values["title"],
                "source": values["source"],
                "url": values["url"],
                "author": values["author"],
                "missing_fields": missing,
                "gap_count": len(missing),
            }
        )

    return {
        "totals": {"result_count": len(result_list), **counts},
        "result_gaps": rows,
    }


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str) -> Iterable[Any]:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value

    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING:
            yield value
        metadata = _field_value(unit, "metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get(key, _MISSING)
            if value is not _MISSING:
                yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, Mapping):
        return None
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        parts = [_string(item) for item in value]
        text = "; ".join(part for part in parts if part)
        return text or None
    text = " ".join(str(value).strip().split())
    return text or None


def _field_text(result: Any, field: str) -> str | None:
    for key in _FIELD_KEYS[field]:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text is not None:
                return text
    return None


def _has_field_value(field: str, value: str | None) -> bool:
    if value is None:
        return False
    if field == "source" and value.casefold() in _UNKNOWN_SOURCE_VALUES:
        return False
    return True


def _result_id(result: Any, index: int) -> str:
    for key in _ID_KEYS:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text is not None:
                return text
    return f"result-{index + 1}"
