"""Summarize source compression encodings."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id


def summarize_source_compression_encodings(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    missing = [row for row in rows if not row["encodings"]]
    counts = Counter(encoding for row in rows for encoding in row["encodings"])
    limit = max(0, sample_limit)
    return {
        "total_sources": len(source_list),
        "encoding_counts": dict(sorted(counts.items())),
        "missing_encoding_count": len(missing),
        "samples": sorted((row for row in rows if row["encodings"]), key=lambda row: sort_key(row["source_id"]))[:limit],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = _encoding(source)
    encodings = [part.strip().casefold() for part in value.split(",") if part.strip()]
    return {"source_id": source_id(source) or str(index), "encodings": encodings}


def _encoding(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    for key in ("content_encoding", "content-encoding", "Content-Encoding"):
        value = field_value(get(source, key)) or field_value(data.get(key))
        if value:
            return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == "content-encoding":
                    return field_value(value)
    return ""
