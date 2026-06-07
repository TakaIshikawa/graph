"""Summarize Cache-Status headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "cache-status"


def summarize_source_cache_status_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    entries = [entry for row in rows for entry in row["entries"]]
    present = [row for row in rows if row["value"]]
    samples = sorted([row for row in rows if row["entries"]], key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    return {
        "total_sources": len(source_list),
        "sources_with_cache_status": len(present),
        "missing_cache_status_count": len(source_list) - len(present),
        "cache_counts": dict(sorted(Counter(entry["cache"] for entry in entries if entry["cache"] and not entry["malformed"]).items())),
        "bucket_counts": {
            "hit": sum(1 for entry in entries if entry["hit"]),
            "miss": sum(1 for entry in entries if entry["fwd"] in {"miss", "uri-miss"}),
            "pass": sum(1 for entry in entries if entry["fwd"] == "bypass"),
            "stale": sum(1 for entry in entries if entry["fwd"] == "stale"),
        },
        "ttl_bucket_counts": _ttl_buckets(entries),
        "collapsed_forwarding_count": sum(1 for entry in entries if entry["collapsed"]),
        "detail_counts": dict(sorted(Counter(entry["detail"] for entry in entries if entry["detail"]).items())),
        "malformed_entry_count": sum(1 for entry in entries if entry["malformed"]),
        "rows": rows,
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    value = field_value(_lookup_header(source, _HEADER))
    return {"source_id": source_id(source) or str(index), "value": value, "entries": [_parse_entry(part) for part in value.split(",") if part.strip()]}


def _parse_entry(text: str) -> dict[str, Any]:
    malformed = text.strip().startswith(";")
    parts = [part.strip() for part in text.split(";") if part.strip()]
    cache = parts[0] if parts else ""
    param_parts = parts if malformed else parts[1:]
    params: dict[str, str] = {}
    for part in param_parts:
        key, _, value = part.partition("=")
        params[key.casefold()] = value.strip('"').casefold() if value else "true"
    return {
        "cache": cache,
        "hit": "hit" in params,
        "fwd": params.get("fwd", ""),
        "ttl": _int_or_none(params.get("ttl", "")),
        "collapsed": params.get("collapsed") == "true",
        "detail": params.get("detail", ""),
        "malformed": malformed or not cache,
    }


def _ttl_buckets(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"expired": 0, "short": 0, "medium": 0, "long": 0}
    for entry in entries:
        ttl = entry["ttl"]
        if ttl is None:
            continue
        if ttl <= 0:
            counts["expired"] += 1
        elif ttl < 60:
            counts["short"] += 1
        elif ttl < 3600:
            counts["medium"] += 1
        else:
            counts["long"] += 1
    return counts


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> Any:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title(), "Cache-Status"):
            value = get(container, key) if container_name == "source" else container.get(key)
            if field_value(value):
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return value
    return ""
