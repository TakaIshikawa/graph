"""Summarize source redirect hints."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_ORIGINAL_KEYS = ("original_url", "url", "source_url", "fetch_url")
_FINAL_KEYS = ("final_url", "redirected_url", "redirect_url", "redirect_target", "canonical_url")


def summarize_source_redirect_hints(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source) for source in source_list]
    hinted = [row for row in rows if _has_redirect_hint(row)]
    redirected = [row for row in hinted if row["original_url"] and row["final_url"] and row["original_url"] != row["final_url"]]
    counts = Counter(row["status_code"] for row in hinted if row["status_code"])
    redirect_counts = [_int(row["redirect_count"]) for row in hinted]
    limit = max(0, sample_limit)
    return {
        "total_sources": len(source_list),
        "sources_with_redirect_hints": len(hinted),
        "redirected_source_count": len(redirected),
        "max_redirect_count": max(redirect_counts, default=0),
        "status_code_counts": dict(sorted(counts.items())),
        "samples": sorted(hinted, key=lambda row: sort_key(row["source_id"]))[:limit],
    }


def _row(source: Mapping[str, Any] | object) -> dict[str, Any]:
    data = metadata(source)
    original = _first(source, data, _ORIGINAL_KEYS)
    final = _first(source, data, _FINAL_KEYS) or original
    return {
        "source_id": source_id(source),
        "original_url": original,
        "final_url": final,
        "redirect_count": _first_value(source, data, "redirect_count"),
        "status_code": _first_value(source, data, "status_code"),
    }


def _first(source: Mapping[str, Any] | object, data: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(source, key))
        if value:
            return value
    for key in keys:
        value = field_value(data.get(key))
        if value:
            return value
    return ""


def _first_value(source: Mapping[str, Any] | object, data: Mapping[str, Any], key: str) -> str:
    value = get(source, key)
    if value is not None:
        return field_value(value)
    return field_value(data.get(key))


def _has_redirect_hint(row: Mapping[str, Any]) -> bool:
    count = field_value(row["redirect_count"])
    if count and count != "0":
        return True
    return bool(row["original_url"] and row["final_url"] and row["original_url"] != row["final_url"])


def _int(value: object) -> int:
    try:
        return int(field_value(value) or "0")
    except ValueError:
        return 0
