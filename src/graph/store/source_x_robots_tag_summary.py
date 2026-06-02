"""Summarize X-Robots-Tag headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "x-robots-tag"


def summarize_source_x_robots_tags(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    directive_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    sources_with = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        raw = _lookup_header(source, _HEADER)
        directives = _directives(raw)
        if not directives:
            continue
        sources_with += 1
        directive_counts.update(directives)
        for directive in directives:
            if len(samples) < limit:
                samples.append({"source_id": sid, "directive": directive, "raw": raw})

    return {
        "total_sources": len(source_list),
        "sources_with_x_robots_tag": sources_with,
        "missing_x_robots_tag_count": len(source_list) - sources_with,
        "directive_counts": {key: directive_counts[key] for key in sorted(directive_counts, key=sort_key)},
        "samples": samples,
    }


def _directives(value: str) -> list[str]:
    directives: list[str] = []
    for part in [field_value(part) for part in value.split(",") if field_value(part)]:
        if ":" in part:
            part = part.split(":", 1)[1]
        name = part.split("=", 1)[0].strip().casefold()
        if name:
            directives.append(name)
    return directives


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
