"""Summarize Report-To headers in sources."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "report-to"


def summarize_source_report_to_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    samples = [
        {"source_id": row["source_id"], "value": row["value"], "malformed": row["malformed"]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_report_to": len(present),
        "group_counts": dict(sorted(Counter(group for row in present for group in row["groups"]).items())),
        "endpoint_host_counts": dict(sorted(Counter(host for row in present for host in row["hosts"]).items())),
        "malformed_count": sum(1 for row in present if row["malformed"]),
        "missing_report_to_count": len(source_list) - len(present),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = _lookup_header(source, _HEADER)
    groups, hosts, malformed = _parse(value)
    return {"source_id": source_id(source) or str(index), "value": value, "groups": groups, "hosts": hosts, "malformed": malformed}


def _parse(value: object) -> tuple[list[str], list[str], bool]:
    text = field_value(value)
    if not text:
        return [], [], False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [], [], True
    items = parsed if isinstance(parsed, list) else [parsed]
    groups: list[str] = []
    hosts: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            return groups, hosts, True
        group = field_value(item.get("group"))
        if group:
            groups.append(group)
        endpoints = item.get("endpoints")
        if isinstance(endpoints, list):
            for endpoint in endpoints:
                if isinstance(endpoint, Mapping):
                    host = urlparse(field_value(endpoint.get("url"))).hostname or ""
                    if host:
                        hosts.append(host)
    return groups, hosts, False


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
