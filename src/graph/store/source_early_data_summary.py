"""Summarize Early-Data headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "early-data"


def summarize_source_early_data_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    value_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    replay_risk_count = unexpected_value_count = sources_with = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        normalized = field_value(value)
        sources_with += 1
        value_counts[normalized] += 1
        if normalized == "1":
            replay_risk_count += 1
        elif normalized not in {"0"}:
            unexpected_value_count += 1
        if len(samples) < limit:
            samples.append({"source_id": sid, "value": normalized})

    return {
        "total_sources": len(source_list),
        "sources_with_early_data": sources_with,
        "missing_early_data_count": len(source_list) - sources_with,
        "value_counts": {key: value_counts[key] for key in sorted(value_counts, key=sort_key)},
        "replay_risk_count": replay_risk_count,
        "unexpected_value_count": unexpected_value_count,
        "samples": samples,
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
