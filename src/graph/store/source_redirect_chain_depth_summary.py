"""Summarize redirect chain depths on source metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_CHAIN_KEYS = ("redirect_chain", "redirects")
_COUNT_KEYS = ("redirect_count", "redirect_depth")
_URL_KEYS = ("url", "source_url", "original_url")
_FINAL_KEYS = ("final_url", "resolved_url", "canonical_url")


def summarize_source_redirect_chain_depths(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    source_list = list(sources)
    buckets: Counter[str] = Counter()
    longest: list[dict[str, Any]] = []
    missing: list[str] = []
    final_domain_changes = 0
    for source in source_list:
        depth = _depth(source)
        sid = source_id(source)
        if depth is None:
            missing.append(sid)
            continue
        buckets[str(depth)] += 1
        longest.append({"source_id": sid, "redirect_depth": depth})
        if _domain_changed(source):
            final_domain_changes += 1
    longest.sort(key=lambda row: (-row["redirect_depth"], sort_key(row["source_id"])))
    return {
        "source_count": len(source_list),
        "depth_buckets": dict(sorted(buckets.items(), key=lambda item: int(item[0]))),
        "longest_chains": longest[:limit],
        "final_domain_change_count": final_domain_changes,
        "sources_missing_redirect_depth": sorted(missing, key=sort_key),
    }


def _depth(source: Any) -> int | None:
    meta = metadata(source)
    for key in _COUNT_KEYS:
        raw = get(source, key, meta.get(key))
        if raw not in (None, ""):
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                return None
    for key in _CHAIN_KEYS:
        raw = get(source, key, meta.get(key))
        if isinstance(raw, list | tuple):
            return len(raw)
        if isinstance(raw, str) and raw.strip():
            return len([part for part in raw.split("->") if part.strip()])
    return None


def _url(source: Any, keys: tuple[str, ...]) -> str:
    meta = metadata(source)
    for key in keys:
        value = field_value(get(source, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""


def _domain_changed(source: Any) -> bool:
    original = urlparse(_url(source, _URL_KEYS)).hostname or ""
    final = urlparse(_url(source, _FINAL_KEYS)).hostname or ""
    return bool(original and final and original.casefold() != final.casefold())
