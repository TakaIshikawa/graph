"""Summarize Via headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "via"


def summarize_source_via_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    hop_distribution: Counter[str] = Counter()
    protocol_counts: Counter[str] = Counter()
    host_counts: Counter[str] = Counter()
    malformed_samples: list[dict[str, str]] = []
    sources_with = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        hops = [field_value(part) for part in value.split(",") if field_value(part)]
        hop_distribution[str(len(hops))] += 1
        for hop in hops:
            parts = hop.split()
            if len(parts) < 2:
                if len(malformed_samples) < limit:
                    malformed_samples.append({"source_id": sid, "value": hop})
                continue
            protocol_counts[parts[0].casefold()] += 1
            host_counts[parts[1].casefold()] += 1

    return {
        "total_sources": len(source_list),
        "sources_with_via": sources_with,
        "sources_missing_via": len(source_list) - sources_with,
        "proxy_hop_count_distribution": {key: hop_distribution[key] for key in sorted(hop_distribution, key=sort_key)},
        "protocol_counts": {key: protocol_counts[key] for key in sorted(protocol_counts, key=sort_key)},
        "top_via_hosts": {key: host_counts[key] for key in sorted(host_counts, key=lambda key: (-host_counts[key], sort_key(key)))[:10]},
        "malformed_via_samples": malformed_samples,
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
