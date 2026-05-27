"""Detect duplicate source references in RAG context items."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse, urlunparse

from graph.rag._analysis_utils import result_id, rounded_ratio, string, value


def audit_context_duplicate_sources(context_items: Iterable[Any]) -> dict[str, Any]:
    """Return duplicate groups by source id, normalized URL, then title."""
    items = list(context_items or [])
    buckets: dict[tuple[str, str], list[tuple[int, Any]]] = defaultdict(list)
    assigned: set[int] = set()

    for key_type in ("source_id", "url", "title"):
        for index, item in enumerate(items):
            if index in assigned:
                continue
            key = _key(item, key_type)
            if key:
                buckets[(key_type, key)].append((index, item))
        for key, grouped in list(buckets.items()):
            if key[0] == key_type and len(grouped) > 1:
                assigned.update(index for index, _ in grouped)

    groups = [
        {
            "key_type": key_type,
            "key": key,
            "item_ids": [result_id(item, index) for index, item in grouped],
            "duplicate_count": len(grouped),
        }
        for (key_type, key), grouped in buckets.items()
        if len(grouped) > 1
    ]
    groups.sort(key=lambda group: (-group["duplicate_count"], group["key_type"], group["key"]))
    duplicate_item_count = sum(group["duplicate_count"] for group in groups)

    return {
        "context_count": len(items),
        "duplicate_group_count": len(groups),
        "duplicate_item_count": duplicate_item_count,
        "groups": groups,
        "diversity_ratio": rounded_ratio(len(items) - duplicate_item_count, len(items)),
    }


def _key(item: Any, key_type: str) -> str | None:
    if key_type == "source_id":
        return string(value(item, "source_id"))
    if key_type == "url":
        text = string(value(item, "url"))
        return _normalize_url(text) if text else None
    return (string(value(item, "title")) or "").casefold() or None


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower() or "https"
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    return urlunparse((scheme, host, path, "", "", ""))
