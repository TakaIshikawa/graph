"""Summarize API key placement hints on source records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_FIELD_KEYS = ("url", "endpoint", "headers", "auth", "authorization", "api_key", "params", "query", "body", "description", "notes")
_API_KEY_RE = re.compile(
    r"\b(?:api[_-]?key|x-api-key|authorization|bearer\s+token|bearer|token|env(?:ironment)?\s+var|request\s+body)\b",
    re.I,
)


def summarize_source_api_key_locations(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = api_key_sources = insecure_query = 0
    location_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []

    for source in sources:
        total += 1
        hints = _hints(source)
        api_key_sources += bool(hints)
        insecure_query += any(location == "query" for _, location, _ in hints)
        for field, location, value in hints:
            location_counts[location] += 1
            if len(samples) < limit:
                samples.append({"source_id": source_id(source), "field": field, "location": location, "value": value})

    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["field"]), sort_key(row["location"])))
    return {
        "total_sources": total,
        "api_key_source_count": api_key_sources,
        "location_counts": {key: location_counts[key] for key in sorted(location_counts, key=sort_key)},
        "insecure_query_count": insecure_query,
        "samples": samples[:limit],
    }


def _hints(source: Any) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field, value in _values(source):
        text = field_value(value)
        haystack = f"{field} {text}"
        if not _API_KEY_RE.search(haystack):
            continue
        for location in _locations(field, text):
            key = (field, location)
            if key not in seen:
                rows.append((field, location, text))
                seen.add(key)
    return rows


def _locations(field: str, text: str) -> list[str]:
    haystack = f"{field} {text}".casefold()
    locations: list[str] = []
    if "bearer" in haystack:
        locations.append("bearer")
    if "authorization" in haystack or "x-api-key" in haystack or "header" in haystack or "headers" in haystack:
        locations.append("header")
    if "query" in haystack or "param" in haystack or re.search(r"[?&](?:api[_-]?key|key|token)=", haystack):
        locations.append("query")
    if "env" in haystack or "environment variable" in haystack:
        locations.append("env")
    if "body" in haystack or "payload" in haystack:
        locations.append("body")
    return locations or ["unspecified"]


def _values(source: Any) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    if isinstance(source, Mapping):
        values.extend(_walk(source))
    else:
        values.extend((key, get(source, key)) for key in _FIELD_KEYS)
    values.extend((f"metadata.{key}", value) for key, value in metadata(source).items())
    return values


def _walk(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, item in value.items():
        field = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            rows.extend(_walk(item, field))
        else:
            rows.append((field, item))
    return rows
