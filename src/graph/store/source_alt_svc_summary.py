"""Summarize Alt-Svc headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "alt-svc"


def summarize_source_alt_svc_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    protocols = Counter(protocol for row in present for protocol in row["protocols"])
    samples = [
        {"source_id": row["source_id"], "protocol": (row["protocols"] or [row["directive"]])[0], "value": row["value"]}
        for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_alt_svc": len(present),
        "protocol_counts": dict(sorted(protocols.items())),
        "clear_count": sum(1 for row in present if row["directive"] == "clear"),
        "missing_alt_svc_count": len(source_list) - len(present),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = _lookup_header(source, _HEADER)
    return {"source_id": source_id(source) or str(index), "value": value, "protocols": _protocols(value), "directive": field_value(value).casefold()}


def _protocols(value: object) -> list[str]:
    protocols = []
    for part in field_value(value).split(","):
        token = part.strip().split("=", 1)[0].strip().casefold()
        if token and token != "clear":
            protocols.append(token)
    return protocols


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
