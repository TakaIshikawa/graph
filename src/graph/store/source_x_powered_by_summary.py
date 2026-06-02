"""Summarize X-Powered-By disclosure headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_TECHNOLOGIES = {"php": "PHP", "express": "Express", "asp.net": "ASP.NET"}


def summarize_source_x_powered_by(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    values: Counter[str] = Counter()
    technologies: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    limit = max(0, sample_limit)
    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, "x-powered-by")
        if not value:
            continue
        values[value] += 1
        folded = value.casefold()
        for token, label in _TECHNOLOGIES.items():
            if token in folded:
                technologies[label] += 1
        if len(samples) < limit:
            samples.append({"source_id": sid, "value": value})
    samples.sort(key=lambda row: sort_key(row["source_id"]))
    present = sum(values.values())
    return {
        "total_sources": len(source_list),
        "sources_with_x_powered_by": present,
        "missing_x_powered_by_count": len(source_list) - present,
        "value_counts": {key: values[key] for key in sorted(values, key=sort_key)},
        "technology_counts": {key: technologies[key] for key in sorted(technologies, key=sort_key)},
        "samples": samples[:limit],
    }


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container in (source, data, get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
