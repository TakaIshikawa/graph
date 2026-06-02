"""Summarize Digest and Content-Digest headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADERS = ("content-digest", "digest")


def summarize_source_content_digests(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    algorithm_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    sources_with = invalid = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        raw = _lookup_any_header(source)
        if not raw:
            continue
        sources_with += 1
        for part in [field_value(part) for part in raw.split(",") if field_value(part)]:
            if "=" not in part:
                invalid += 1
                if len(samples) < limit:
                    samples.append({"source_id": sid, "algorithm": "", "value": part, "raw": raw})
                continue
            algorithm, value = part.split("=", 1)
            algorithm = field_value(algorithm).casefold()
            value = field_value(value).strip("\"'")
            if not algorithm or not value:
                invalid += 1
                continue
            algorithm_counts[algorithm] += 1
            if len(samples) < limit:
                samples.append({"source_id": sid, "algorithm": algorithm, "value": value, "raw": raw})

    return {
        "total_sources": len(source_list),
        "sources_with_content_digest": sources_with,
        "missing_content_digest_count": len(source_list) - sources_with,
        "invalid_digest_count": invalid,
        "algorithm_counts": {key: algorithm_counts[key] for key in sorted(algorithm_counts, key=sort_key)},
        "samples": samples,
    }


def _lookup_any_header(source: Mapping[str, Any] | object) -> str:
    for header in _HEADERS:
        value = _lookup_header(source, header)
        if value:
            return value
    return ""


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
