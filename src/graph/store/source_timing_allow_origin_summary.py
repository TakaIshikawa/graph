"""Summarize Timing-Allow-Origin headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, sort_key, source_id

_HEADER = "timing-allow-origin"


def summarize_source_timing_allow_origins(
    sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5
) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    origin_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    sources_with = wildcard_origin_count = multi_origin_source_count = empty_value_count = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        found, origins = _lookup_origins(source)
        if not found:
            continue
        if not origins:
            empty_value_count += 1
            continue

        sources_with += 1
        origin_counts.update(origins)
        if "*" in origins:
            wildcard_origin_count += 1
        if len(origins) > 1:
            multi_origin_source_count += 1
        samples.append({"source_id": sid, "origins": origins})

    samples.sort(key=lambda row: sort_key(row["source_id"]))
    return {
        "total_sources": len(source_list),
        "sources_with_timing_allow_origin": sources_with,
        "missing_header_count": len(source_list) - sources_with - empty_value_count,
        "empty_value_count": empty_value_count,
        "wildcard_origin_count": wildcard_origin_count,
        "explicit_origin_count": sum(count for origin, count in origin_counts.items() if origin != "*"),
        "multi_origin_source_count": multi_origin_source_count,
        "origin_counts": {key: origin_counts[key] for key in sorted(origin_counts, key=sort_key)},
        "samples": samples[:limit],
    }


def _lookup_origins(source: Mapping[str, Any] | object) -> tuple[bool, list[str]]:
    values = _lookup_header_values(source, _HEADER)
    if not values:
        return False, []
    origins = sorted(
        {
            origin
            for value in values
            for origin in (field_value(part) for part in field_value(value).split(","))
            if origin
        },
        key=sort_key,
    )
    return True, origins


def _lookup_header_values(source: Mapping[str, Any] | object, header: str) -> list[object]:
    data = metadata(source)
    values: list[object] = []
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            raw = get(container, key) if container_name == "source" else container.get(key)
            if raw is not None:
                values.extend(flatten_values(raw))
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    values.extend(flatten_values(value))
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    values.extend(flatten_values(value))
    return values
