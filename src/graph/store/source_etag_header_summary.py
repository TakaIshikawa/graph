"""Summarize ETag headers in sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "etag"


def summarize_source_etag_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    rows_by_etag: dict[str, dict[str, Any]] = {}
    sources_with = weak_count = strong_count = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        etag = _lookup_header(source, _HEADER)
        if not etag:
            continue

        sources_with += 1
        validator_type = _validator_type(etag)
        if validator_type == "weak":
            weak_count += 1
        else:
            strong_count += 1

        row = rows_by_etag.setdefault(
            etag,
            {"etag": etag, "validator_type": validator_type, "count": 0, "source_ids": [], "examples": []},
        )
        row["count"] += 1
        if sid not in row["source_ids"] and len(row["source_ids"]) < limit:
            row["source_ids"].append(sid)
        if etag not in row["examples"] and len(row["examples"]) < limit:
            row["examples"].append(etag)

    rows = sorted(rows_by_etag.values(), key=lambda row: (sort_key(row["etag"]), sort_key(row["validator_type"])))
    return {
        "total_sources": len(source_list),
        "sources_with_etag": sources_with,
        "missing_etag_count": len(source_list) - sources_with,
        "weak_etag_count": weak_count,
        "strong_etag_count": strong_count,
        "distinct_etag_count": len(rows_by_etag),
        "rows": rows,
    }


def _validator_type(value: str) -> str:
    return "weak" if value.lstrip().casefold().startswith("w/") else "strong"


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if _normalized_header_key(key) == header:
                    text = field_value(value)
                    if text:
                        return text
        else:
            for key in (header, header.upper(), header.title(), "e_tag"):
                value = field_value(get(container, key))
                if value:
                    return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if _normalized_header_key(key) == header:
                    text = field_value(value)
                    if text:
                        return text
    return ""


def _normalized_header_key(value: object) -> str:
    return field_value(value).casefold().replace("_", "").replace("-", "")
