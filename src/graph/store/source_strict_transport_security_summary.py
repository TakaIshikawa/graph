"""Summarize Strict-Transport-Security headers in sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "strict-transport-security"


def summarize_source_strict_transport_security(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    rows_sorted = sorted(present, key=lambda row: sort_key(row["source_id"]))
    limit = max(0, sample_limit)
    return {
        "total_sources": len(source_list),
        "sources_with_header": len(present),
        "missing_header_count": len(source_list) - len(present),
        "include_subdomains_count": sum(1 for row in present if row["include_subdomains"]),
        "preload_count": sum(1 for row in present if row["preload"]),
        "missing_max_age_count": sum(1 for row in present if not row["has_max_age"]),
        "invalid_max_age_count": sum(1 for row in present if row["invalid_max_age"]),
        "max_age_seconds": [row["max_age_seconds"] for row in rows_sorted if isinstance(row["max_age_seconds"], int)],
        "max_age_buckets": _buckets(row["max_age_seconds"] for row in present if isinstance(row["max_age_seconds"], int)),
        "source_ids": [row["source_id"] for row in rows_sorted],
        "rows": rows_sorted,
        "samples": [{"source_id": row["source_id"], "max_age_seconds": row["max_age_seconds"], "include_subdomains": row["include_subdomains"], "preload": row["preload"]} for row in rows_sorted[:limit]],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = field_value(_lookup_header(source, _HEADER)).strip()
    directives = _directives(value)
    raw_max_age = directives.get("max-age", "")
    max_age: int | str = ""
    invalid = False
    if raw_max_age:
        try:
            max_age = int(raw_max_age)
            invalid = max_age < 0
        except ValueError:
            invalid = True
    return {
        "source_id": source_id(source) or str(index),
        "value": value,
        "max_age_seconds": max_age,
        "has_max_age": "max-age" in directives,
        "invalid_max_age": invalid,
        "include_subdomains": "includesubdomains" in directives,
        "preload": "preload" in directives,
    }


def _directives(value: str) -> dict[str, str]:
    directives: dict[str, str] = {}
    for part in value.split(";"):
        token = part.strip()
        if not token:
            continue
        if "=" in token:
            key, raw_value = token.split("=", 1)
            directives[key.strip().casefold()] = raw_value.strip().strip('"')
        else:
            directives[token.casefold()] = ""
    return directives


def _buckets(values: Iterable[int]) -> dict[str, int]:
    buckets = {"zero": 0, "lt_1_day": 0, "lt_30_days": 0, "gte_30_days": 0}
    for value in values:
        if value <= 0:
            buckets["zero"] += 1
        elif value < 86400:
            buckets["lt_1_day"] += 1
        elif value < 2592000:
            buckets["lt_30_days"] += 1
        else:
            buckets["gte_30_days"] += 1
    return {key: count for key, count in buckets.items() if count}


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
