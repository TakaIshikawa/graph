"""Summarize CORS response headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADERS = (
    "access-control-allow-origin",
    "access-control-allow-credentials",
    "access-control-allow-methods",
    "access-control-allow-headers",
)


def summarize_source_cors_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["headers"]]
    origin_counts = Counter(row["headers"].get("access-control-allow-origin", "") for row in present)
    origin_counts.pop("", None)
    samples = [
        {"source_id": row["source_id"], "headers": row["headers"]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_cors_headers": len(present),
        "allow_origin_counts": dict(sorted(origin_counts.items())),
        "credentialed_count": sum(1 for row in present if row["headers"].get("access-control-allow-credentials", "").casefold() == "true"),
        "wildcard_origin_count": sum(1 for row in present if row["headers"].get("access-control-allow-origin") == "*"),
        "missing_header_count": len(source_list) - len(present),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    return {"source_id": source_id(source) or str(index), "headers": _lookup_headers(source)}


def _lookup_headers(source: Mapping[str, Any] | object) -> dict[str, str]:
    return {header: value for header in _HEADERS if (value := _lookup_header(source, header))}


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    aliases = (header, header.replace("-", "_"), header.title())
    for container_name, container in (("source", source), ("metadata", data)):
        for key in aliases:
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
