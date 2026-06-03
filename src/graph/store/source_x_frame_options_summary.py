"""Summarize X-Frame-Options headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "x-frame-options"
_KNOWN = {"deny", "sameorigin", "allow-from"}


def summarize_source_x_frame_options(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    unknown = [row for row in present if row["policy"] not in _KNOWN]
    rows_sorted = sorted(present, key=lambda row: sort_key(row["source_id"]))
    limit = max(0, sample_limit)
    return {
        "total_sources": len(source_list),
        "sources_with_header": len(present),
        "missing_header_count": len(source_list) - len(present),
        "policy_counts": {key: count for key, count in sorted(Counter(row["policy"] for row in present if row["policy"] in _KNOWN).items(), key=lambda item: sort_key(item[0]))},
        "unknown_value_count": len(unknown),
        "unknown_values": _examples(unknown, "value", limit),
        "source_ids": [row["source_id"] for row in rows_sorted],
        "rows": rows_sorted,
        "samples": [{"source_id": row["source_id"], "policy": row["policy"], "value": row["value"]} for row in rows_sorted[:limit]],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    value = field_value(_lookup_header(source, _HEADER)).strip()
    return {"source_id": source_id(source) or str(index), "value": value, "policy": _normalize(value)}


def _normalize(value: str) -> str:
    token = value.strip().split(None, 1)[0].casefold() if value.strip() else ""
    return "allow-from" if token == "allow-from" else token


def _examples(rows: list[dict[str, str]], field: str, limit: int) -> list[dict[str, Any]]:
    counts = Counter(row[field] for row in rows)
    first_ids: dict[str, list[str]] = {}
    for row in sorted(rows, key=lambda item: sort_key(item["source_id"])):
        first_ids.setdefault(row[field], []).append(row["source_id"])
    return [
        {"value": value, "count": counts[value], "source_ids": first_ids[value][:limit]}
        for value in sorted(counts, key=sort_key)[:limit]
    ]


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
