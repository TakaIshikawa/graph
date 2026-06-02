"""Summarize Content-Location headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "content-location"


def summarize_source_content_location_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    domains: Counter[str] = Counter()
    invalid_samples: list[dict[str, str]] = []
    sources_with = absolute = relative = same_domain = cross_domain = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            absolute += 1
            host = parsed.hostname or parsed.netloc.casefold()
            domains[host.casefold()] += 1
            source_host = _source_host(source)
            if source_host and source_host == host.casefold():
                same_domain += 1
            elif source_host:
                cross_domain += 1
        elif not parsed.scheme and not parsed.netloc and value.startswith(("/", "./", "../")):
            relative += 1
        else:
            if len(invalid_samples) < limit:
                invalid_samples.append({"source_id": sid, "value": value})

    return {
        "total_sources": len(source_list),
        "sources_with_content_location": sources_with,
        "sources_missing_content_location": len(source_list) - sources_with,
        "absolute_url_count": absolute,
        "relative_url_count": relative,
        "same_domain_count": same_domain,
        "cross_domain_count": cross_domain,
        "top_content_location_domains": {key: domains[key] for key in sorted(domains, key=lambda key: (-domains[key], sort_key(key)))[:10]},
        "invalid_content_location_samples": invalid_samples,
    }


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
