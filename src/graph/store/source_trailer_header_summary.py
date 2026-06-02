"""Summarize Trailer headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "trailer"
_COMMON = {"digest", "signature", "content-md5", "etag"}


def summarize_source_trailer_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    field_counts: Counter[str] = Counter()
    unusual_samples: list[dict[str, str]] = []
    sources_with = digest = signature = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        for field in [field_value(part).casefold().replace("_", "-") for part in value.split(",") if field_value(part)]:
            field_counts[field] += 1
            digest += field == "digest"
            signature += field == "signature"
            if field not in _COMMON and len(unusual_samples) < limit:
                unusual_samples.append({"source_id": sid, "field": field})

    return {
        "total_sources": len(source_list),
        "sources_with_trailer": sources_with,
        "sources_missing_trailer": len(source_list) - sources_with,
        "trailer_field_counts": {key: field_counts[key] for key in sorted(field_counts, key=sort_key)},
        "digest_trailer_count": digest,
        "signature_trailer_count": signature,
        "unusual_trailer_samples": unusual_samples,
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
