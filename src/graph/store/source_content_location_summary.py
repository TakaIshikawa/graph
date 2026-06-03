"""Summarize Content-Location headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "content-location"


def summarize_source_content_locations(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    hostname_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    invalid_samples: list[dict[str, str]] = []
    sources_with = same_domain = cross_domain = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        kind, hostname = _classify_content_location(value)
        kind_counts[kind] += 1
        if hostname:
            hostname_counts[hostname] += 1
            source_host = _source_host(source)
            if source_host and source_host == hostname:
                same_domain += 1
            elif source_host:
                cross_domain += 1
        if kind == "other" and len(invalid_samples) < limit:
            invalid_samples.append({"source_id": sid, "value": value})
        if len(samples) < limit:
            samples.append({"source_id": sid, "kind": kind, "value": value})

    return {
        "total_sources": len(source_list),
        "sources_with_content_location": sources_with,
        "missing_content_location_count": len(source_list) - sources_with,
        "sources_missing_content_location": len(source_list) - sources_with,
        "kind_counts": {key: kind_counts[key] for key in sorted(kind_counts, key=sort_key)},
        "hostname_counts": {key: hostname_counts[key] for key in sorted(hostname_counts, key=lambda key: (-hostname_counts[key], sort_key(key)))},
        "absolute_url_count": kind_counts["absolute_url"],
        "relative_url_count": kind_counts["root_relative_path"] + kind_counts["relative_path"],
        "same_domain_count": same_domain,
        "cross_domain_count": cross_domain,
        "top_content_location_domains": {
            key: hostname_counts[key] for key in sorted(hostname_counts, key=lambda key: (-hostname_counts[key], sort_key(key)))[:10]
        },
        "invalid_content_location_samples": invalid_samples,
        "samples": samples,
    }


def summarize_source_content_location_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    return summarize_source_content_locations(sources, sample_limit)


def _classify_content_location(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return "absolute_url", (parsed.hostname or parsed.netloc).casefold()
    if not parsed.scheme and not parsed.netloc and value.startswith("/"):
        return "root_relative_path", ""
    if not parsed.scheme and not parsed.netloc and value.startswith(("./", "../")):
        return "relative_path", ""
    if not parsed.scheme and not parsed.netloc and parsed.path and not any(char.isspace() for char in value):
        return "relative_path", ""
    return "other", ""


def _source_host(source: Mapping[str, Any] | object) -> str:
    for key in ("url", "source_url", "uri"):
        parsed = urlparse(field_value(get(source, key) or metadata(source).get(key)))
        if parsed.hostname:
            return parsed.hostname.casefold()
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
