"""Summarize Accept-CH client hint headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "accept-ch"


def summarize_source_accept_ch_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    hint_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    sources_with = empty_value_count = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        found, raw = _lookup_header(source, _HEADER)
        if not found:
            continue
        hints = sorted({field_value(part).casefold() for part in raw.split(",") if field_value(part)}, key=sort_key)
        if not hints:
            empty_value_count += 1
            continue
        sources_with += 1
        hint_counts.update(hints)
        if len(samples) < limit:
            samples.append({"source_id": sid, "hints": hints})

    samples.sort(key=lambda row: sort_key(row["source_id"]))
    return {
        "total_sources": len(source_list),
        "sources_with_accept_ch": sources_with,
        "missing_header_count": len(source_list) - sources_with - empty_value_count,
        "hint_counts": {key: hint_counts[key] for key in sorted(hint_counts, key=sort_key)},
        "empty_value_count": empty_value_count,
        "samples": samples[:limit],
    }


def summarize_source_accept_ch_hints(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    return summarize_source_accept_ch_headers(sources, sample_limit)


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> tuple[bool, str]:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            raw = get(container, key) if container_name == "source" else container.get(key)
            if raw is not None:
                return True, field_value(raw)
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return True, field_value(value)
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return True, field_value(value)
    return False, ""
