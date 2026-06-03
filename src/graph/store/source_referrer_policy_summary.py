"""Summarize Referrer-Policy headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "referrer-policy"
_KNOWN = {
    "no-referrer",
    "no-referrer-when-downgrade",
    "origin",
    "origin-when-cross-origin",
    "same-origin",
    "strict-origin",
    "strict-origin-when-cross-origin",
    "unsafe-url",
}


def summarize_source_referrer_policies(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["tokens"]]
    rows_sorted = sorted(present, key=lambda row: sort_key(row["source_id"]))
    invalid_rows = [row for row in present if row["invalid_tokens"]]
    limit = max(0, sample_limit)
    return {
        "total_sources": len(source_list),
        "sources_with_header": len(present),
        "missing_header_count": len(source_list) - len(present),
        "effective_policy_counts": _counter(row["effective_policy"] for row in present if row["effective_policy"]),
        "token_counts": _counter(token for row in present for token in row["valid_tokens"]),
        "invalid_token_count": sum(len(row["invalid_tokens"]) for row in present),
        "invalid_values": _invalid_examples(invalid_rows, limit),
        "source_ids": [row["source_id"] for row in rows_sorted],
        "rows": rows_sorted,
        "samples": [{"source_id": row["source_id"], "effective_policy": row["effective_policy"], "tokens": row["tokens"]} for row in rows_sorted[:limit]],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = field_value(_lookup_header(source, _HEADER)).strip()
    tokens = [part.strip().casefold() for part in value.split(",") if part.strip()]
    valid = [token for token in tokens if token in _KNOWN]
    invalid = [token for token in tokens if token not in _KNOWN]
    return {
        "source_id": source_id(source) or str(index),
        "value": value,
        "tokens": tokens,
        "valid_tokens": valid,
        "invalid_tokens": invalid,
        "effective_policy": valid[-1] if valid else "",
    }


def _counter(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts, key=sort_key)}


def _invalid_examples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    counts = Counter(token for row in rows for token in row["invalid_tokens"])
    source_ids: dict[str, list[str]] = {}
    for row in sorted(rows, key=lambda item: sort_key(item["source_id"])):
        for token in row["invalid_tokens"]:
            source_ids.setdefault(token, []).append(row["source_id"])
    return [{"value": token, "count": counts[token], "source_ids": source_ids[token][:limit]} for token in sorted(counts, key=sort_key)[:limit]]


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
