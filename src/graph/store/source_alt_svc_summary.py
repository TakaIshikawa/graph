"""Summarize Alt-Svc headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "alt-svc"


def summarize_source_alt_svcs(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    protocol_counts: Counter[str] = Counter()
    max_age_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    max_age_samples: list[dict[str, str]] = []
    sample_order = 0
    sources_with = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        entries = _entries(value)
        for entry in entries:
            protocol_counts[entry["protocol"]] += 1
            if entry.get("ma"):
                max_age_counts[entry["ma"]] += 1
            sample = {"source_id": sid, "protocol": entry["protocol"], "value": entry["raw"], "_order": sample_order}
            sample_order += 1
            if entry.get("ma"):
                sample["ma"] = entry["ma"]
                max_age_samples.append({"source_id": sid, "protocol": entry["protocol"], "ma": entry["ma"], "value": entry["raw"]})
            samples.append(sample)

    samples.sort(key=lambda row: (sort_key(row["source_id"]), row["_order"]))
    bounded_samples = samples[:limit]
    for sample in bounded_samples:
        sample.pop("_order", None)
    max_age_samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["protocol"]), sort_key(row["ma"]), sort_key(row["value"])))

    return {
        "total_sources": len(source_list),
        "sources_with_alt_svc": sources_with,
        "missing_alt_svc_count": len(source_list) - sources_with,
        "protocol_counts": {key: protocol_counts[key] for key in sorted(protocol_counts, key=sort_key)},
        "max_age_counts": {key: max_age_counts[key] for key in sorted(max_age_counts, key=sort_key)},
        "max_age_samples": max_age_samples[:limit],
        "clear_count": protocol_counts["clear"],
        "samples": bounded_samples,
    }


def summarize_source_alt_svc_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = []
    for index, source in enumerate(source_list):
        value = _lookup_header(source, _HEADER)
        entries = _entries(value)
        rows.append({"source_id": source_id(source) or str(index), "value": value, "entries": entries})
    present = [row for row in rows if row["value"]]
    protocols = Counter(entry["protocol"] for row in present for entry in row["entries"] if entry["protocol"] != "clear")
    samples = []
    for row in sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]:
        protocol = row["entries"][0]["protocol"] if row["entries"] else field_value(row["value"]).casefold()
        samples.append({"source_id": row["source_id"], "protocol": protocol, "value": row["value"]})
    return {
        "total_sources": len(source_list),
        "sources_with_alt_svc": len(present),
        "protocol_counts": dict(sorted(protocols.items())),
        "clear_count": sum(1 for row in present for entry in row["entries"] if entry["protocol"] == "clear"),
        "missing_alt_svc_count": len(source_list) - len(present),
        "samples": samples,
    }


def _entries(value: object) -> list[dict[str, str]]:
    entries = []
    for part in _split_quoted(field_value(value), ","):
        raw = part.strip()
        if not raw:
            continue
        token, sep, rest = raw.partition("=")
        protocol = field_value(token).casefold()
        if protocol == "clear":
            entries.append({"protocol": "clear", "raw": raw})
            continue
        if not protocol or not sep:
            continue
        entry = {"protocol": protocol, "raw": raw}
        for param in _split_quoted(rest, ";")[1:]:
            key, param_sep, param_value = param.partition("=")
            if param_sep and field_value(key).casefold() == "ma":
                ma = _unquote(param_value)
                if ma:
                    entry["ma"] = ma
        entries.append(entry)
    return entries


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


def _unquote(value: object) -> str:
    text = field_value(value)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1].replace(r"\"", '"')
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
