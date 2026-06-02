"""Summarize X-Permitted-Cross-Domain-Policies headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "x-permitted-cross-domain-policies"
_KNOWN = {"none", "master-only", "by-content-type", "by-ftp-filename", "all"}


def summarize_source_x_permitted_cross_domain_policies(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["policy"]]
    noteworthy = [row for row in present if row["policy"] == "all" or row["policy"] not in _KNOWN]
    samples = [
        {"source_id": row["source_id"], "policy": row["policy"]}
        for row in sorted(noteworthy, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_header": len(present),
        "policy_counts": dict(sorted(Counter(row["policy"] for row in present).items())),
        "missing_header_count": len(source_list) - len(present),
        "permissive_count": sum(1 for row in present if row["policy"] == "all"),
        "unknown_value_count": sum(1 for row in present if row["policy"] not in _KNOWN),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    return {"source_id": source_id(source) or str(index), "policy": field_value(_lookup_header(source, _HEADER)).casefold()}


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
