"""Summarize Service-Worker-Allowed response headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id


def summarize_source_service_worker_allowed(
    sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5
) -> dict[str, Any]:
    source_list = list(sources)
    scopes: Counter[str] = Counter()
    broad = 0
    samples: list[dict[str, str]] = []
    limit = max(0, sample_limit)
    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        scope = _normalize(_lookup_header(source, "service-worker-allowed"))
        if not scope:
            continue
        scopes[scope] += 1
        if scope == "/" or scope.endswith("://") or scope.count("/") <= 2 and "://" in scope:
            broad += 1
        if len(samples) < limit:
            samples.append({"source_id": sid, "scope": scope})
    samples.sort(key=lambda row: sort_key(row["source_id"]))
    present = sum(scopes.values())
    return {
        "total_sources": len(source_list),
        "sources_with_service_worker_allowed": present,
        "missing_service_worker_allowed_count": len(source_list) - present,
        "scope_counts": {key: scopes[key] for key in sorted(scopes, key=sort_key)},
        "broad_scope_count": broad,
        "samples": samples[:limit],
    }


def _normalize(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        return value.rstrip("/") + ("/" if value.count("/") <= 2 else "")
    return "/" + value.lstrip("/") if not value.startswith("/") else value


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container in (source, data, get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
