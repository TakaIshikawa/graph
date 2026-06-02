"""Summarize Expect-CT headers in sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "expect-ct"


def summarize_source_expect_ct_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    samples = [
        {"source_id": row["source_id"], "value": row["value"], "max_age": row["max_age"], "enforce": row["enforce"]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_expect_ct": len(present),
        "enforce_count": sum(1 for row in present if row["enforce"]),
        "report_uri_count": sum(1 for row in present if row["report_uri"]),
        "missing_max_age_count": sum(1 for row in present if not row["has_max_age"]),
        "invalid_max_age_count": sum(1 for row in present if row["invalid_max_age"]),
        "missing_header_count": len(source_list) - len(present),
        "max_age_buckets": _buckets(row["max_age"] for row in present if isinstance(row["max_age"], int)),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = _lookup_header(source, _HEADER)
    directives = _directives(value)
    max_age = directives.get("max-age", "")
    parsed_max_age: int | str = ""
    invalid = False
    if max_age:
        try:
            parsed_max_age = int(max_age)
            invalid = parsed_max_age < 0
        except ValueError:
            invalid = True
    return {
        "source_id": source_id(source) or str(index),
        "value": value,
        "max_age": parsed_max_age,
        "has_max_age": "max-age" in directives,
        "invalid_max_age": invalid,
        "enforce": "enforce" in directives,
        "report_uri": field_value(directives.get("report-uri")),
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
