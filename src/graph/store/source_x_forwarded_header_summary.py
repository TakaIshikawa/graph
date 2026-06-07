"""Summarize X-Forwarded headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from ipaddress import ip_address
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADERS = {
    "for": "x-forwarded-for",
    "proto": "x-forwarded-proto",
    "host": "x-forwarded-host",
    "port": "x-forwarded-port",
}


def summarize_source_x_forwarded_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if any(row[name] for name in _HEADERS)]
    samples = sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    return {
        "total_sources": len(source_list),
        "sources_with_x_forwarded": len(present),
        "header_presence_counts": {name: sum(1 for row in rows if row[name]) for name in _HEADERS if any(row[name] for row in rows)},
        "proto_counts": dict(sorted(Counter(row["proto"] for row in rows if row["proto"]).items())),
        "max_hop_count": max((row["hop_count"] for row in rows), default=0),
        "private_address_hint_count": sum(1 for row in rows if row["private_address_hint"]),
        "missing_x_forwarded_count": len(source_list) - len(present),
        "rows": sorted(present, key=lambda row: sort_key(row["source_id"])),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    values = {name: field_value(_lookup_header(source, header)) for name, header in _HEADERS.items()}
    forwarded_for = values["for"]
    hops = [part.strip() for part in forwarded_for.split(",") if part.strip()]
    return {
        "source_id": source_id(source) or str(index),
        "for": forwarded_for,
        "proto": values["proto"].casefold(),
        "host": values["host"],
        "port": values["port"],
        "hop_count": len(hops),
        "private_address_hint": any(_private_address_hint(hop) for hop in hops),
    }


def _private_address_hint(value: str) -> bool:
    text = value.strip().strip("[]")
    if not text:
        return False
    host = text.rsplit(":", 1)[0] if text.count(":") == 1 else text
    try:
        parsed = ip_address(host.strip("[]"))
    except ValueError:
        return False
    return parsed.is_private or parsed.is_loopback or parsed.is_link_local


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
