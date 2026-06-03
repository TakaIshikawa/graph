"""Summarize Link headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import re
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "link"
_REL_RE = re.compile(r"^[a-z0-9_.:-]+$")


def summarize_source_link_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    rel_counts: Counter[str] = Counter()
    rows_by_rel: dict[str, dict[str, Any]] = {}
    sources_with = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        for rel in _rels(value):
            rel_counts[rel] += 1
            row = rows_by_rel.setdefault(rel, {"rel": rel, "count": 0, "source_ids": [], "examples": []})
            row["count"] += 1
            if sid not in row["source_ids"] and len(row["source_ids"]) < limit:
                row["source_ids"].append(sid)
            if value not in row["examples"] and len(row["examples"]) < limit:
                row["examples"].append(value)

    return {
        "total_sources": len(source_list),
        "sources_with_link_header": sources_with,
        "missing_link_header_count": len(source_list) - sources_with,
        "rel_counts": {key: rel_counts[key] for key in sorted(rel_counts, key=sort_key)},
        "rows": [rows_by_rel[key] for key in sorted(rows_by_rel, key=sort_key)],
    }


def _rels(value: str) -> list[str]:
    rels: list[str] = []
    for entry in _split_link_entries(value):
        for param in _split_quoted(entry, ";")[1:]:
            key, sep, raw_value = param.partition("=")
            if not sep or field_value(key).casefold() != "rel":
                continue
            for rel in _unquote(raw_value).split():
                normalized = rel.casefold()
                if _REL_RE.fullmatch(normalized):
                    rels.append(normalized)
    return rels


def _split_link_entries(value: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    quote = False
    escape = False
    angle_depth = 0
    for char in value:
        if escape:
            buf.append(char)
            escape = False
            continue
        if char == "\\" and quote:
            buf.append(char)
            escape = True
            continue
        if char == '"':
            quote = not quote
        elif char == "<" and not quote:
            angle_depth += 1
        elif char == ">" and not quote and angle_depth:
            angle_depth -= 1
        if char == "," and not quote and angle_depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    parts.append("".join(buf))
    return [part.strip() for part in parts if part.strip()]


def _split_quoted(value: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    quote = False
    escape = False
    for char in value:
        if escape:
            buf.append(char)
            escape = False
            continue
        if char == "\\" and quote:
            buf.append(char)
            escape = True
            continue
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
