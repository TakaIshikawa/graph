"""Summarize Forwarded headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from ipaddress import ip_address
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "forwarded"
_PARAMS = ("by", "for", "host", "proto")


def summarize_source_forwarded_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    parameter_presence_counts: Counter[str] = Counter()
    proto_value_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    sources_with = obfuscated_identifier_count = private_identifier_count = malformed_count = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        for element in _elements(value):
            params = element["params"]
            if not params:
                malformed_count += 1
                continue
            for key in _PARAMS:
                if key in params:
                    parameter_presence_counts[key] += 1
            if params.get("proto"):
                proto_value_counts[params["proto"].casefold()] += 1
            for key in ("by", "for"):
                identifier = params.get(key, "")
                if identifier.startswith("_"):
                    obfuscated_identifier_count += 1
                elif _private_identifier(identifier):
                    private_identifier_count += 1
            if len(samples) < limit:
                samples.append({"source_id": sid, "params": params, "value": element["raw"]})

    return {
        "total_sources": len(source_list),
        "sources_with_forwarded": sources_with,
        "parameter_presence_counts": {key: parameter_presence_counts[key] for key in _PARAMS if parameter_presence_counts[key]},
        "proto_value_counts": {key: proto_value_counts[key] for key in sorted(proto_value_counts, key=sort_key)},
        "obfuscated_identifier_count": obfuscated_identifier_count,
        "private_identifier_count": private_identifier_count,
        "malformed_count": malformed_count,
        "missing_forwarded_count": len(source_list) - sources_with,
        "samples": samples,
    }


def _elements(value: str) -> list[dict[str, Any]]:
    rows = []
    for element in _split_quoted(value, ","):
        raw = element.strip()
        if not raw:
            continue
        params = {}
        malformed = False
        for part in _split_quoted(raw, ";"):
            key, sep, raw_value = part.partition("=")
            if not sep:
                malformed = True
                continue
            name = field_value(key).casefold()
            if name:
                params[name] = _unquote(raw_value)
        rows.append({"params": {} if malformed and not params else params, "raw": raw})
    return rows


def _private_identifier(value: str) -> bool:
    text = value.strip("[]")
    if not text:
        return False
    host = text.rsplit(":", 1)[0] if text.count(":") == 1 else text
    try:
        parsed = ip_address(host.strip("[]"))
    except ValueError:
        return False
    return parsed.is_private or parsed.is_loopback or parsed.is_link_local


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
