"""Summarize Digest headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "digest"
_STRONG = {"sha-256", "sha-384", "sha-512"}
_WEAK = {"md5", "sha", "sha-1"}


def summarize_source_digest_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    algorithm_counts: Counter[str] = Counter()
    invalid_samples: list[dict[str, str]] = []
    sources_with = strong = weak = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        for part in [field_value(part) for part in value.split(",") if field_value(part)]:
            if "=" not in part:
                if len(invalid_samples) < limit:
                    invalid_samples.append({"source_id": sid, "value": part})
                continue
            algorithm = part.split("=", 1)[0].strip().casefold()
            if not algorithm:
                if len(invalid_samples) < limit:
                    invalid_samples.append({"source_id": sid, "value": part})
                continue
            algorithm_counts[algorithm] += 1
            strong += algorithm in _STRONG
            weak += algorithm in _WEAK

    return {
        "total_sources": len(source_list),
        "sources_with_digest": sources_with,
        "sources_missing_digest": len(source_list) - sources_with,
        "algorithm_counts": {key: algorithm_counts[key] for key in sorted(algorithm_counts, key=sort_key)},
        "strong_digest_count": strong,
        "weak_or_legacy_digest_count": weak,
        "invalid_digest_samples": invalid_samples,
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
