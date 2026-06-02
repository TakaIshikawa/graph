"""Summarize HTTP method hints in source records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, normalized_key, sort_key, source_id

_METHOD_KEYS = {"method", "http_method", "request_method"}
_NESTED_KEYS = ("request", "options")
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def summarize_source_http_method_hints(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["method"]]
    method_counts = Counter(row["method"] for row in present)
    samples = [
        {"source_id": row["source_id"], "method": row["method"], "field": row["field"]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_method_hint": len(present),
        "method_counts": dict(sorted(method_counts.items())),
        "non_get_count": sum(1 for row in present if row["method"] != "GET"),
        "unsafe_method_count": sum(1 for row in present if row["method"] in _UNSAFE_METHODS),
        "missing_method_hint_count": len(source_list) - len(present),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    method, field = _method(source)
    return {"source_id": source_id(source) or str(index), "method": method, "field": field}


def _method(source: Mapping[str, Any] | object) -> tuple[str, str]:
    for key in ("method", "http_method", "request_method"):
        text = field_value(get(source, key))
        if text:
            return text.upper(), key
    found = _find_method(metadata(source), "metadata")
    if found[0]:
        return found
    for key in _NESTED_KEYS:
        value = get(source, key)
        if isinstance(value, Mapping):
            found = _find_method(value, key)
            if found[0]:
                return found
    return "", ""


def _find_method(values: Mapping[str, Any], prefix: str) -> tuple[str, str]:
    for key, value in values.items():
        key_text = field_value(key)
        norm = normalized_key(key_text)
        if norm in _METHOD_KEYS and field_value(value):
            return field_value(value).upper(), f"{prefix}.{key_text}"
        if norm in _NESTED_KEYS and isinstance(value, Mapping):
            found = _find_method(value, f"{prefix}.{key_text}")
            if found[0]:
                return found
    return "", ""
