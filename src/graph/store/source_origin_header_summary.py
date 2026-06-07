"""Summarize Origin headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "origin"


def summarize_source_origin_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["origin"]]
    samples = sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    return {
        "total_sources": len(source_list),
        "sources_with_origin": len(present),
        "missing_origin_count": len(source_list) - len(present),
        "origin_counts": dict(sorted(Counter(row["origin"] for row in present).items())),
        "scheme_counts": dict(sorted(Counter(row["scheme"] for row in present if row["scheme"]).items())),
        "domain_counts": dict(sorted(Counter(row["host"] for row in present if row["host"]).items())),
        "null_origin_count": sum(1 for row in present if row["origin"] == "null"),
        "opaque_file_origin_count": sum(1 for row in present if row["scheme"] == "file" or row["origin"] in {"null", "opaque"}),
        "host_mismatch_count": sum(1 for row in present if row["source_host"] and row["host"] and row["source_host"] != row["host"]),
        "rows": sorted(present, key=lambda row: sort_key(row["source_id"])),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    origin = field_value(_lookup_header(source, _HEADER))
    parsed = urlparse(origin)
    source_host = urlparse(field_value(get(source, "url")) or field_value(metadata(source).get("url"))).netloc.casefold()
    return {
        "source_id": source_id(source) or str(index),
        "origin": origin,
        "scheme": parsed.scheme.casefold(),
        "host": parsed.netloc.casefold(),
        "source_host": source_host,
    }


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> Any:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = get(container, key) if container_name == "source" else container.get(key)
            if field_value(value):
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return value
    return ""
