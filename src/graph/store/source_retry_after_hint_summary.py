"""Summarize source Retry-After hints."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_RETRY_AFTER_KEYS = ("retry_after", "retry-after", "Retry-After", "Retry_After", "retryAfter")


def summarize_source_retry_after_hints(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize Retry-After style metadata and response headers."""
    rows = [_row(source) for source in sources]
    hinted = [row for row in rows if row["value"]]
    type_counts = Counter(row["value_type"] for row in hinted)
    hinted.sort(key=lambda row: sort_key(row["source_id"]))
    limit = max(0, sample_limit)
    return {
        "sources_with_retry_after_hint": len(hinted),
        "retry_after_count": len(hinted),
        "value_type_counts": {key: type_counts[key] for key in sorted(type_counts, key=sort_key)},
        "samples": hinted[:limit],
    }


def _row(source: Mapping[str, Any] | object) -> dict[str, str]:
    value = _retry_after(source)
    return {"source_id": source_id(source), "value": value, "value_type": _value_type(value)}


def _retry_after(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    for key in _RETRY_AFTER_KEYS:
        value = field_value(get(source, key) or data.get(key))
        if value:
            return value
    headers = get(source, "headers") or data.get("headers") or data.get("response_headers") or {}
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if str(key).casefold() == "retry-after":
                return field_value(value)
    return ""


def _value_type(value: str) -> str:
    if not value:
        return ""
    if value.isdecimal():
        return "seconds"
    try:
        parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return "invalid"
    return "http-date"
