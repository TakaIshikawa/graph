"""Summarize Pragma headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "pragma"


def summarize_source_pragma_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    value_counts: Counter[str] = Counter()
    unusual_samples: list[dict[str, str]] = []
    sources_with = no_cache = other = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        normalized = value.casefold()
        value_counts[normalized] += 1
        if normalized == "no-cache":
            no_cache += 1
        else:
            other += 1
            if len(unusual_samples) < limit:
                unusual_samples.append({"source_id": sid, "value": normalized})

    return {
        "total_sources": len(source_list),
        "sources_with_pragma": sources_with,
        "sources_missing_pragma": len(source_list) - sources_with,
        "no_cache_count": no_cache,
        "other_value_count": other,
        "top_pragma_values": {key: value_counts[key] for key in sorted(value_counts, key=lambda key: (-value_counts[key], sort_key(key)))[:10]},
        "unusual_pragma_samples": unusual_samples,
    }


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
