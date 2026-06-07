"""Summarize Sec-Fetch-User headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "sec-fetch-user"


def summarize_source_sec_fetch_users(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    samples = sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    return {
        "total_sources": len(source_list),
        "sources_with_sec_fetch_user": len(present),
        "missing_sec_fetch_user_count": len(source_list) - len(present),
        "value_counts": dict(sorted(Counter(row["normalized"] for row in present).items())),
        "user_activation_count": sum(1 for row in present if row["normalized"] == "?1"),
        "non_user_activation_count": sum(1 for row in present if row["normalized"] == "?0"),
        "invalid_or_blank_count": sum(1 for row in rows if row["raw_value"] and not row["normalized"]),
        "unexpected_value_count": sum(1 for row in present if row["normalized"] not in {"?0", "?1"}),
        "rows": sorted(present, key=lambda row: sort_key(row["source_id"])),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    raw_value = _lookup_header(source, _HEADER)
    value = field_value(raw_value)
    normalized = value.casefold()
    return {"source_id": source_id(source) or str(index), "raw_value": str(raw_value) if raw_value is not None else "", "value": value, "normalized": normalized}


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> Any:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title(), "Sec-Fetch-User"):
            value = get(container, key) if container_name == "source" else container.get(key)
            if field_value(value):
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return value
    return ""
