"""Summarize OAuth scopes on source records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_SCOPE_KEYS = ("scope", "scopes", "oauth_scopes")
_BROAD_RE = re.compile(r"(write|delete|admin|full)", re.IGNORECASE)


def summarize_source_oauth_scopes(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    scope_counts: Counter[str] = Counter()
    sources_with_scopes = 0
    sources_without_scopes = 0
    broad_scope_samples: list[dict[str, str]] = []
    source_samples: list[dict[str, Any]] = []

    for source in sources:
        sid = source_id(source)
        scopes = _scopes(source)
        if scopes:
            sources_with_scopes += 1
            scope_counts.update(scopes)
            if len(source_samples) < limit:
                source_samples.append({"source_id": sid, "scopes": scopes})
        else:
            sources_without_scopes += 1
        for scope in scopes:
            if _BROAD_RE.search(scope) and len(broad_scope_samples) < limit:
                broad_scope_samples.append({"source_id": sid, "scope": scope})

    return {
        "scope_counts": {key: scope_counts[key] for key in sorted(scope_counts, key=sort_key)},
        "sources_with_scopes": sources_with_scopes,
        "sources_without_scopes": sources_without_scopes,
        "broad_scope_samples": sorted(broad_scope_samples, key=lambda row: (sort_key(row["source_id"]), sort_key(row["scope"]))),
        "source_samples": sorted(source_samples, key=lambda row: sort_key(row["source_id"])),
    }


def _scopes(source: Any) -> list[str]:
    meta = metadata(source)
    for key in _SCOPE_KEYS:
        raw = get(source, key)
        if raw not in (None, ""):
            return _normalize(raw)
        raw = meta.get(key)
        if raw not in (None, ""):
            return _normalize(raw)
    return []


def _normalize(value: Any) -> list[str]:
    values = value if isinstance(value, list | tuple | set) else re.split(r"[\s,]+", field_value(value))
    scopes = {field_value(scope).strip().casefold() for scope in values}
    return sorted((scope for scope in scopes if scope), key=sort_key)
