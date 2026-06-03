"""Summarize Content-Encoding headers in sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "content-encoding"
_KNOWN_ENCODINGS = {"br", "compress", "deflate", "gzip", "identity", "zstd"}


def summarize_source_content_encodings(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows_by_encoding: dict[str, dict[str, Any]] = {}
    sources_with = 0
    limit = max(0, sample_limit)

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        encodings = _encodings(value)
        if not encodings:
            continue
        sources_with += 1
        for encoding in encodings:
            row = rows_by_encoding.setdefault(
                encoding,
                {"encoding": encoding, "count": 0, "known": encoding in _KNOWN_ENCODINGS, "source_ids": [], "examples": []},
            )
            row["count"] += 1
            if sid not in row["source_ids"] and len(row["source_ids"]) < limit:
                row["source_ids"].append(sid)
            if value not in row["examples"] and len(row["examples"]) < limit:
                row["examples"].append(value)

    rows = sorted(rows_by_encoding.values(), key=lambda row: sort_key(row["encoding"]))
    return {
        "total_sources": len(source_list),
        "sources_with_content_encoding": sources_with,
        "missing_content_encoding_count": len(source_list) - sources_with,
        "unknown_encoding_count": sum(row["count"] for row in rows if not row["known"]),
        "rows": rows,
        "encoding_counts": {row["encoding"]: row["count"] for row in rows},
    }


def _encodings(value: str) -> list[str]:
    return [encoding for encoding in (field_value(part).casefold() for part in value.split(",")) if encoding]


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
