"""Summarize source HTTP cache headers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER_KEYS = ("cache-control", "etag", "last-modified", "expires", "age")
_ALIASES = {key: {key, key.replace("-", "_"), key.title()} for key in _HEADER_KEYS}


def summarize_source_cache_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize HTTP cache header coverage and Cache-Control directives."""
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["headers"]]
    directives = Counter(
        directive
        for row in present
        for directive in _cache_directives(row["headers"].get("cache-control", ""))
    )
    limit = max(0, sample_limit)
    samples = [
        {"source_id": row["source_id"], "headers": row["headers"]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[:limit]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_cache_headers": len(present),
        "directive_counts": dict(sorted(directives.items())),
        "missing_header_count": len(source_list) - len(present),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    data = metadata(source)
    headers: dict[str, str] = {}
    for header in _HEADER_KEYS:
        value = _header_value(source, data, header)
        if value:
            headers[header] = value
    return {"source_id": source_id(source) or str(index), "headers": headers}


def _header_value(source: Mapping[str, Any] | object, data: Mapping[str, Any], header: str) -> str:
    for key in _ALIASES[header]:
        value = field_value(get(source, key))
        if value:
            return value
    for key in _ALIASES[header]:
        value = field_value(data.get(key))
        if value:
            return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""


def _cache_directives(value: object) -> list[str]:
    directives = []
    for part in field_value(value).split(","):
        name = part.strip().split("=", 1)[0].strip().casefold()
        if name:
            directives.append(name)
    return directives
