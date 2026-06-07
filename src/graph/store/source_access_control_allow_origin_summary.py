"""Summarize Access-Control-Allow-Origin headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_ORIGIN_HEADER = "access-control-allow-origin"
_CREDENTIALS_HEADER = "access-control-allow-credentials"


def summarize_source_access_control_allow_origins(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    samples = sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    return {
        "total_sources": len(source_list),
        "sources_with_origin": len(present),
        "missing_origin_count": len(source_list) - len(present),
        "value_counts": dict(sorted(Counter(row["value"] for row in present).items())),
        "wildcard_count": sum(1 for row in present if row["value"] == "*"),
        "null_origin_count": sum(1 for row in present if row["value"] == "null"),
        "credential_wildcard_conflict_count": sum(1 for row in present if row["value"] == "*" and row["allow_credentials"] == "true"),
        "origin_echo_count": sum(1 for row in present if row["value"] == row["request_origin"] and row["request_origin"]),
        "invalid_or_blank_count": sum(1 for row in rows if row["raw_value"] and not row["value"]),
        "domain_counts": dict(sorted(Counter(row["domain"] for row in present if row["domain"]).items())),
        "rows": sorted(present, key=lambda row: sort_key(row["source_id"])),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    raw_value = _lookup_header(source, _ORIGIN_HEADER)
    value = field_value(raw_value)
    return {
        "source_id": source_id(source) or str(index),
        "raw_value": str(raw_value) if raw_value is not None else "",
        "value": value,
        "domain": urlparse(value).netloc.casefold() if "://" in value else "",
        "allow_credentials": field_value(_lookup_header(source, _CREDENTIALS_HEADER)).casefold(),
        "request_origin": field_value(_lookup_header(source, "origin")),
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
