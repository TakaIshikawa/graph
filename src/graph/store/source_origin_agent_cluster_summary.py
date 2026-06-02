"""Summarize Origin-Agent-Cluster headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "origin-agent-cluster"


def summarize_source_origin_agent_clusters(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    problematic = [row for row in present if row["state"] in {"disabled", "unknown"}]
    samples = [
        {"source_id": row["source_id"], "value": row["value"], "state": row["state"]}
        for row in sorted(problematic, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_header": len(present),
        "enabled_count": sum(1 for row in present if row["state"] == "enabled"),
        "disabled_count": sum(1 for row in present if row["state"] == "disabled"),
        "unknown_value_count": sum(1 for row in present if row["state"] == "unknown"),
        "missing_header_count": len(source_list) - len(present),
        "value_counts": dict(sorted(Counter(row["value"] for row in present).items())),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    value = field_value(_lookup_header(source, _HEADER)).casefold()
    return {"source_id": source_id(source) or str(index), "value": value, "state": _state(value)}


def _state(value: str) -> str:
    if value in {"?1", "true", "1"}:
        return "enabled"
    if value in {"?0", "false", "0"}:
        return "disabled"
    return "unknown"


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
