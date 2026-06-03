"""Summarize Access-Control-Expose-Headers headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "access-control-expose-headers"


def summarize_source_access_control_expose_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    counts: Counter[str] = Counter()
    wildcard_count = 0
    for row in present:
        for header in row["headers"]:
            if header == "*":
                wildcard_count += 1
            else:
                counts[header] += 1
    rows_sorted = sorted(present, key=lambda row: sort_key(row["source_id"]))
    return {
        "total_sources": len(source_list),
        "sources_with_access_control_expose_headers": len(present),
        "missing_access_control_expose_headers_count": len(source_list) - len(present),
        "exposed_header_counts": dict(sorted(counts.items(), key=lambda item: sort_key(item[0]))),
        "wildcard_count": wildcard_count,
        "rows": rows_sorted,
        "samples": rows_sorted[: max(0, sample_limit)],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = _lookup_header(source, _HEADER)
    headers = [field_value(part).casefold() for part in value.split(",") if field_value(part)]
    return {"source_id": source_id(source) or str(index), "value": value, "headers": headers}


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
