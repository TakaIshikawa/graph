"""Audit canonical URL coverage and duplicates in RAG results."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse, urlunparse

from graph.rag._analysis_utils import result_id, string, value

_URL_KEYS = ("canonical_url", "url", "source_url", "link")


def audit_result_canonical_urls(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return missing and duplicate canonical URL groups."""
    missing = []
    groups: dict[str, list[dict[str, str]]] = {}
    for index, item in enumerate(results or []):
        raw = _raw_url(item)
        normalized = _normalize_url(raw)
        if normalized is None:
            missing.append({"item_id": result_id(item, index), "index": index})
            continue
        groups.setdefault(normalized, []).append({"item_id": result_id(item, index), "url": raw or ""})
    duplicates = [
        {"canonical_url": url, "items": items}
        for url, items in sorted(groups.items())
        if len(items) > 1
    ]
    return {
        "missing_url_items": missing,
        "duplicate_url_groups": duplicates,
        "total_results": len(results or []),
        "unique_url_count": len(groups),
    }


def _raw_url(item: Any) -> str | None:
    for key in _URL_KEYS:
        text = string(value(item, key))
        if text:
            return text
    return None


def _normalize_url(raw: str | None) -> str | None:
    text = string(raw)
    if text is None:
        return None
    parsed = urlparse(text)
    scheme = (parsed.scheme or "https").casefold()
    host = parsed.netloc.casefold()
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((scheme, host, path, "", parsed.query, ""))
