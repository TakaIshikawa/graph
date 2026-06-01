"""Summarize canonical URL conflicts on source records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse, urlunparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

CANONICAL_KEYS = ("canonical_url", "canonical", "rel_canonical")
URL_KEYS = ("url", "final_url", "source_url")


def summarize_source_canonical_url_conflicts(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = with_canonical = 0
    canonical_to_sources: dict[str, set[str]] = defaultdict(set)
    source_to_canonicals: dict[str, set[str]] = defaultdict(set)
    samples: list[dict[str, str]] = []

    for source in sources:
        total += 1
        source_url = _source_url(source)
        canonical_url = _canonical_url(source)
        if not canonical_url:
            continue
        with_canonical += 1
        normalized_source = _normalize_url(source_url)
        normalized_canonical = _normalize_url(canonical_url)
        canonical_to_sources[normalized_canonical].add(normalized_source)
        source_to_canonicals[normalized_source].add(normalized_canonical)
        if len(samples) < limit:
            samples.append({"source_id": source_id(source), "url": normalized_source, "canonical_url": normalized_canonical})

    canonical_groups = [
        {"type": "canonical", "canonical_url": canonical, "urls": sorted(urls, key=sort_key), "count": len(urls)}
        for canonical, urls in canonical_to_sources.items()
        if len(urls) > 1
    ]
    url_groups = [
        {"type": "url", "url": url, "canonical_urls": sorted(canonicals, key=sort_key), "count": len(canonicals)}
        for url, canonicals in source_to_canonicals.items()
        if len(canonicals) > 1
    ]
    canonical_groups.sort(key=lambda row: (-row["count"], sort_key(row["canonical_url"])))
    url_groups.sort(key=lambda row: (-row["count"], sort_key(row["url"])))
    samples.sort(key=lambda row: (sort_key(row["url"]), sort_key(row["canonical_url"]), sort_key(row["source_id"])))
    return {
        "total_sources": total,
        "sources_with_canonical_url": with_canonical,
        "canonical_conflict_count": len(canonical_groups),
        "url_conflict_count": len(url_groups),
        "conflict_groups": [*canonical_groups, *url_groups],
        "samples": samples[:limit],
    }


def _source_url(source: Any) -> str:
    meta = metadata(source)
    for key in URL_KEYS:
        value = get(source, key)
        if field_value(value):
            return field_value(value)
    for key in URL_KEYS:
        value = meta.get(key)
        if field_value(value):
            return field_value(value)
    return ""


def _canonical_url(source: Any) -> str:
    meta = metadata(source)
    for key in CANONICAL_KEYS:
        value = meta.get(key)
        if field_value(value):
            return field_value(value)
    for key in CANONICAL_KEYS:
        value = get(source, key)
        if field_value(value):
            return field_value(value)
    return ""


def _normalize_url(value: Any) -> str:
    text = field_value(value)
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else f"https://{text}")
    scheme = (parsed.scheme or "https").casefold()
    host = (parsed.hostname or "").casefold()
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))
