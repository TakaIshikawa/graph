"""Summarize Cross-Origin-Embedder-Policy headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "cross-origin-embedder-policy"
_KNOWN = {"unsafe-none", "require-corp", "credentialless"}
_ISOLATING = {"require-corp", "credentialless"}


def summarize_source_cross_origin_embedder_policies(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["policy"]]
    unknown = [row for row in present if row["policy"] not in _KNOWN]
    noteworthy = [row for row in present if row["policy"] == "unsafe-none" or row["policy"] not in _KNOWN]
    rows_sorted = sorted(present, key=lambda row: sort_key(row["source_id"]))
    limit = max(0, sample_limit)
    return {
        "total_sources": len(source_list),
        "sources_with_policy": len(present),
        "policy_counts": dict(sorted(Counter(row["policy"] for row in present if row["policy"] in _KNOWN).items())),
        "missing_policy_count": len(source_list) - len(present),
        "isolating_policy_count": sum(1 for row in present if row["policy"] in _ISOLATING),
        "weak_policy_count": sum(1 for row in present if row["policy"] == "unsafe-none"),
        "unknown_value_count": len(unknown),
        "unknown_values": _unknown_values(unknown, limit),
        "source_ids": [row["source_id"] for row in rows_sorted],
        "rows": rows_sorted,
        "samples": [{"source_id": row["source_id"], "policy": row["policy"], "field": row["field"]} for row in sorted(noteworthy, key=lambda row: sort_key(row["source_id"]))[:limit]],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    value, field = _lookup_header(source, _HEADER)
    return {"source_id": source_id(source) or str(index), "policy": field_value(value).casefold(), "field": field}


def _unknown_values(rows: list[dict[str, str]], limit: int) -> list[dict[str, Any]]:
    counts = Counter(row["policy"] for row in rows)
    source_ids: dict[str, list[str]] = {}
    for row in sorted(rows, key=lambda item: sort_key(item["source_id"])):
        source_ids.setdefault(row["policy"], []).append(row["source_id"])
    return [{"value": value, "count": counts[value], "source_ids": source_ids[value][:limit]} for value in sorted(counts, key=sort_key)[:limit]]


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> tuple[str, str]:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value, key
    for owner, container in (("headers", get(source, "headers")), ("response_headers", get(source, "response_headers")), ("metadata.headers", data.get("headers")), ("metadata.response_headers", data.get("response_headers"))):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value), f"{owner}.{key}"
    return "", ""
