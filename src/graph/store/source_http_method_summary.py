"""Summarize HTTP methods used by source records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

METHOD_KEYS = ("method", "http_method", "request_method", "fetch_method")
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def summarize_source_http_methods(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = unsafe = missing = 0
    method_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []

    for source in sources:
        total += 1
        field, raw_value = _method_value(source)
        method = field_value(raw_value).upper()
        if not method:
            missing += 1
            continue
        method_counts[method] += 1
        unsafe += method in UNSAFE_METHODS
        if len(samples) < limit:
            samples.append({"source_id": source_id(source), "field": field or "", "method": method})

    samples.sort(key=lambda row: (sort_key(row["method"]), sort_key(row["source_id"])))
    return {
        "total_sources": total,
        "method_counts": {key: method_counts[key] for key in sorted(method_counts, key=sort_key)},
        "unsafe_method_count": unsafe,
        "missing_method_count": missing,
        "samples": samples[:limit],
    }


def _method_value(source: Any) -> tuple[str | None, Any]:
    meta = metadata(source)
    for key in METHOD_KEYS:
        value = meta.get(key)
        if field_value(value):
            return f"metadata.{key}", value
    for key in METHOD_KEYS:
        value = get(source, key)
        if field_value(value):
            return key, value
    return None, None
