"""Summarize Cross-Origin-Resource-Policy headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "cross-origin-resource-policy"
_KNOWN = {"same-origin", "same-site", "cross-origin"}


def summarize_source_cross_origin_resource_policies(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["policy"]]
    invalid = [row for row in present if row["policy"] not in _KNOWN]
    permissive = [row for row in present if row["policy"] == "cross-origin"]
    noteworthy = permissive + invalid
    samples = [
        {"source_id": row["source_id"], "policy": row["policy"]}
        for row in sorted(noteworthy, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    bucket_counts = Counter(row["policy"] if row["policy"] in _KNOWN else "invalid" for row in present)
    bucket_counts["missing"] = len(source_list) - len(present)
    return {
        "total_sources": len(source_list),
        "sources_with_policy": len(present),
        "policy_counts": dict(sorted(Counter(row["policy"] for row in present if row["policy"] in _KNOWN).items())),
        "bucket_counts": {key: bucket_counts[key] for key in ("same-origin", "same-site", "cross-origin", "invalid", "missing")},
        "missing_policy_count": len(source_list) - len(present),
        "permissive_count": len(permissive),
        "invalid_value_count": len(invalid),
        "invalid_values": _invalid_values(invalid, max(0, sample_limit)),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    value = _lookup_header(source, _HEADER)
    return {"source_id": source_id(source) or str(index), "policy": field_value(value).casefold()}


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


def _invalid_values(rows: list[dict[str, str]], limit: int) -> list[dict[str, Any]]:
    counts = Counter(row["policy"] for row in rows)
    source_ids: dict[str, list[str]] = {}
    for row in sorted(rows, key=lambda item: sort_key(item["source_id"])):
        source_ids.setdefault(row["policy"], []).append(row["source_id"])
    return [{"value": value, "count": counts[value], "source_ids": source_ids[value][:limit]} for value in sorted(counts, key=sort_key)[:limit]]
