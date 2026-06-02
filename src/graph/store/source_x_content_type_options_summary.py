"""Summarize X-Content-Type-Options headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "x-content-type-options"


def summarize_source_x_content_type_options(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    unusual = [row for row in present if row["value"] != "nosniff"]
    limit = max(0, sample_limit)
    samples = [
        {"source_id": row["source_id"], "value": row["value"], "field": row["field"]}
        for row in sorted(unusual, key=lambda row: sort_key(row["source_id"]))[:limit]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_header": len(present),
        "value_counts": dict(sorted(Counter(row["value"] for row in present).items())),
        "missing_header_count": len(source_list) - len(present),
        "non_nosniff_count": len(unusual),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    value, field = _header_value(source)
    return {"source_id": source_id(source) or str(index), "value": value.casefold(), "field": field}


def _header_value(source: Mapping[str, Any] | object) -> tuple[str, str]:
    data = metadata(source)
    for owner, container in (("source", source), ("metadata", data)):
        for key in (_HEADER, _HEADER.replace("-", "_"), "X-Content-Type-Options"):
            value = field_value(get(container, key) if owner == "source" else container.get(key))
            if value:
                return value, key
    for owner, container in (("headers", get(source, "headers")), ("response_headers", get(source, "response_headers")), ("metadata.headers", data.get("headers")), ("metadata.response_headers", data.get("response_headers"))):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == _HEADER:
                    return field_value(value), f"{owner}.{key}"
    return "", ""
