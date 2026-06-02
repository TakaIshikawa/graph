"""Summarize Proxy-Status headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "proxy-status"


def summarize_source_proxy_statuses(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    proxy_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    next_hop_protocol_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    sources_with = malformed_entry_count = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        for entry in _entries(value):
            if not entry["proxy"]:
                malformed_entry_count += 1
                continue
            proxy_counts[entry["proxy"]] += 1
            if entry["params"].get("error"):
                error_counts[entry["params"]["error"]] += 1
            if entry["params"].get("next-hop-protocol"):
                next_hop_protocol_counts[entry["params"]["next-hop-protocol"]] += 1
            if len(samples) < limit:
                samples.append({"source_id": sid, "proxy": entry["proxy"], "value": entry["raw"]})

    return {
        "total_sources": len(source_list),
        "sources_with_proxy_status": sources_with,
        "proxy_counts": {key: proxy_counts[key] for key in sorted(proxy_counts, key=sort_key)},
        "error_counts": {key: error_counts[key] for key in sorted(error_counts, key=sort_key)},
        "next_hop_protocol_counts": {key: next_hop_protocol_counts[key] for key in sorted(next_hop_protocol_counts, key=sort_key)},
        "malformed_entry_count": malformed_entry_count,
        "missing_proxy_status_count": len(source_list) - sources_with,
        "samples": samples,
    }


def _entries(value: str) -> list[dict[str, Any]]:
    rows = []
    for entry in _split_quoted(value, ","):
        raw = entry.strip()
        if not raw:
            continue
        parts = _split_quoted(raw, ";")
        proxy = field_value(parts[0]).casefold()
        params = {}
        for part in parts[1:]:
            key, sep, raw_value = part.partition("=")
            if sep:
                params[field_value(key).casefold()] = _unquote(raw_value).casefold()
        rows.append({"proxy": proxy, "params": params, "raw": raw})
    return rows


def _split_quoted(value: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    quote = False
    for char in value:
        if char == '"':
            quote = not quote
        if char == delimiter and not quote:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    parts.append("".join(buf))
    return parts


def _unquote(value: str) -> str:
    text = field_value(value)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1].replace('\\"', '"')
    return text


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
