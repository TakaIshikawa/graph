"""Summarize Transfer-Encoding headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "transfer-encoding"


def summarize_source_transfer_encodings(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    chunked_count = sources_with = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source)
        tokens = [part.strip().casefold() for part in value.split(",") if part.strip()]
        if not tokens:
            continue
        sources_with += 1
        counts.update(tokens)
        chunked_count += int("chunked" in tokens)
        rows.append({"source_id": sid, "transfer_encodings": tokens, "raw": value})

    return {
        "total_sources": len(source_list),
        "sources_with_transfer_encoding": sources_with,
        "missing_transfer_encoding_count": len(source_list) - sources_with,
        "encoding_counts": {key: counts[key] for key in sorted(counts, key=sort_key)},
        "chunked_count": chunked_count,
        "samples": sorted(rows, key=lambda row: sort_key(row["source_id"]))[:limit],
    }


def _lookup_header(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (_HEADER, _HEADER.replace("-", "_"), _HEADER.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == _HEADER:
                    return field_value(value)
    return ""
