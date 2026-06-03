"""Summarize Vary headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "vary"


def summarize_source_vary_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    token_counts: Counter[str] = Counter()
    rows_by_token: dict[str, dict[str, Any]] = {}
    sources_with = wildcard_count = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        raw = _lookup_header(source, _HEADER)
        if not raw:
            continue
        tokens = _tokens(raw)
        if not tokens:
            continue

        sources_with += 1
        wildcard_count += "*" in tokens
        token_counts.update(tokens)
        for token in tokens:
            row = rows_by_token.setdefault(token, {"token": token, "count": 0, "source_ids": [], "examples": []})
            row["count"] += 1
            if sid not in row["source_ids"] and len(row["source_ids"]) < limit:
                row["source_ids"].append(sid)
            if raw not in row["examples"] and len(row["examples"]) < limit:
                row["examples"].append(raw)

    rows = sorted(rows_by_token.values(), key=lambda row: sort_key(row["token"]))
    return {
        "total_sources": len(source_list),
        "sources_with_vary": sources_with,
        "missing_vary_count": len(source_list) - sources_with,
        "wildcard_vary_count": wildcard_count,
        "token_counts": {key: token_counts[key] for key in sorted(token_counts, key=sort_key)},
        "rows": rows,
    }


def _tokens(value: str) -> set[str]:
    return {token for token in (_normalize_token(part) for part in value.split(",")) if token}


def _normalize_token(value: object) -> str:
    return field_value(value).casefold().replace("_", "-")


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
