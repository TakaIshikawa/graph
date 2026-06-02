"""Summarize Cache-Control headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "cache-control"
_COMMON = {"max-age", "no-cache", "no-store", "immutable", "private", "public"}


def summarize_source_cache_control_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    directive_counts: Counter[str] = Counter()
    noteworthy_samples: list[dict[str, str]] = []
    sources_with = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        for directive in _directives(value):
            name = directive.split("=", 1)[0].strip().casefold()
            if not name:
                continue
            directive_counts[name] += 1
            if name not in _COMMON and len(noteworthy_samples) < limit:
                noteworthy_samples.append({"source_id": sid, "directive": directive})

    return {
        "total_sources": len(source_list),
        "sources_with_cache_control": sources_with,
        "missing_cache_control_count": len(source_list) - sources_with,
        "directive_counts": {key: directive_counts[key] for key in sorted(directive_counts, key=sort_key)},
        "noteworthy_samples": noteworthy_samples,
    }


def _directives(value: str) -> list[str]:
    return [field_value(part).casefold() for chunk in value.split(",") for part in chunk.split(";") if field_value(part)]


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
