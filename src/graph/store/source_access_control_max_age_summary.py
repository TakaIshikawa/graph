"""Summarize Access-Control-Max-Age headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "access-control-max-age"


def summarize_source_access_control_max_ages(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    valid = [row for row in present if row["valid"]]
    bucket_counts = Counter(row["bucket"] for row in valid)
    rows_sorted = sorted(present, key=lambda row: sort_key(row["source_id"]))
    return {
        "total_sources": len(source_list),
        "sources_with_access_control_max_age": len(present),
        "missing_access_control_max_age_count": len(source_list) - len(present),
        "bucket_counts": dict(sorted(bucket_counts.items(), key=lambda item: sort_key(item[0]))),
        "invalid_value_count": sum(1 for row in present if not row["valid"]),
        "rows": rows_sorted,
        "samples": rows_sorted[: max(0, sample_limit)],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = _lookup_header(source, _HEADER)
    valid = value.isdecimal()
    seconds = int(value) if valid else None
    return {"source_id": source_id(source) or str(index), "value": value, "seconds": seconds if seconds is not None else "", "bucket": _bucket(seconds) if seconds is not None else "invalid", "valid": valid}


def _bucket(seconds: int) -> str:
    if seconds == 0:
        return "0"
    if seconds <= 600:
        return "1-600"
    if seconds <= 3600:
        return "601-3600"
    if seconds <= 86400:
        return "3601-86400"
    return "86401+"


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
