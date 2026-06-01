"""Summarize source ETag metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_ETAG_KEYS = ("etag", "ETag", "e_tag")


def summarize_source_etags(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize ETag presence, validator strength, and duplicates."""
    source_list = list(sources)
    values: list[tuple[str, str]] = []
    missing = weak = strong = 0
    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        etag = _etag(source)
        if not etag:
            missing += 1
            continue
        values.append((sid, etag))
        if etag.casefold().startswith("w/"):
            weak += 1
        else:
            strong += 1
    counts = Counter(etag for _, etag in values)
    duplicate_etags = [
        {"etag": etag, "count": counts[etag], "source_ids": [sid for sid, value in sorted(values, key=lambda item: sort_key(item[0])) if value == etag][:sample_limit]}
        for etag in sorted(counts, key=sort_key)
        if counts[etag] > 1
    ]
    examples = [{"source_id": sid, "etag": etag} for sid, etag in sorted(values, key=lambda item: (sort_key(item[0]), sort_key(item[1])))[:sample_limit]]
    return {
        "total_sources": len(source_list),
        "sources_with_etag": len(values),
        "sources_missing_etag": missing,
        "weak_etag_count": weak,
        "strong_etag_count": strong,
        "duplicate_etags": duplicate_etags,
        "examples": examples,
    }


def _etag(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    headers = get(source, "headers") or data.get("headers") or data.get("response_headers") or {}
    for key in _ETAG_KEYS:
        value = field_value(get(source, key) or data.get(key))
        if value:
            return value
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if str(key).casefold() == "etag":
                return field_value(value)
    return ""
