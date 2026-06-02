"""Summarize Accept-CH client hint headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "accept-ch"


def summarize_source_accept_ch_hints(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    hint_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    sources_with = total_hints = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        raw = _lookup_header(source, _HEADER)
        hints = sorted({field_value(part).casefold() for part in raw.split(",") if field_value(part)}, key=sort_key) if raw else []
        if not hints:
            continue
        sources_with += 1
        total_hints += len(hints)
        hint_counts.update(hints)
        for hint in hints:
            if len(samples) < limit:
                samples.append({"source_id": sid, "hint": hint, "raw": raw})

    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["hint"])))
    return {
        "total_sources": len(source_list),
        "sources_with_accept_ch": sources_with,
        "missing_accept_ch_count": len(source_list) - sources_with,
        "total_accept_ch_hints": total_hints,
        "hint_counts": {key: hint_counts[key] for key in sorted(hint_counts, key=sort_key)},
        "samples": samples[:limit],
    }


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
