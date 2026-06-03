"""Summarize Reporting-Endpoints headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "reporting-endpoints"


def summarize_source_reporting_endpoints(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["value"]]
    rows_sorted = sorted(present, key=lambda row: sort_key(row["source_id"]))
    endpoint_names = Counter(endpoint["name"] for row in present for endpoint in row["endpoints"])
    return {
        "total_sources": len(source_list),
        "sources_with_reporting_endpoints": len(present),
        "endpoint_name_counts": dict(sorted(endpoint_names.items(), key=lambda item: sort_key(item[0]))),
        "https_endpoint_count": sum(1 for row in present for endpoint in row["endpoints"] if endpoint["scheme"] == "https"),
        "non_https_endpoint_count": sum(1 for row in present for endpoint in row["endpoints"] if endpoint["scheme"] != "https"),
        "malformed_count": sum(row["malformed_count"] for row in present),
        "missing_count": len(source_list) - len(present),
        "rows": rows_sorted,
        "samples": rows_sorted[: max(0, sample_limit)],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = _lookup_header(source, _HEADER)
    endpoints, malformed_count = _parse_reporting_endpoints(value)
    return {
        "source_id": source_id(source) or str(index),
        "value": value,
        "endpoints": endpoints,
        "malformed_count": malformed_count,
    }


def _parse_reporting_endpoints(value: object) -> tuple[list[dict[str, str]], int]:
    text = field_value(value)
    if not text:
        return [], 0
    endpoints: list[dict[str, str]] = []
    malformed_count = 0
    for entry in _split_header_entries(text):
        name, separator, url = entry.partition("=")
        name = field_value(name)
        url = _unquote(field_value(url))
        parsed = urlparse(url)
        if separator != "=" or not name or not url or not parsed.scheme:
            malformed_count += 1
            continue
        endpoints.append({"name": name, "url": url, "scheme": parsed.scheme.casefold()})
    return endpoints, malformed_count


def _split_header_entries(value: str) -> list[str]:
    entries: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for character in value:
        if escaped:
            current.append(character)
            escaped = False
            continue
        if character == "\\" and quote:
            current.append(character)
            escaped = True
            continue
        if character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
        if character == "," and quote is None:
            entry = field_value("".join(current))
            if entry:
                entries.append(entry)
            current = []
            continue
        current.append(character)
    entry = field_value("".join(current))
    if entry:
        entries.append(entry)
    return entries


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


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
