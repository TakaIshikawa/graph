"""Store summary for unit canonical URL coverage."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_URL_KEYS = ("canonical_url", "url", "source_url", "origin_url", "permalink")


def summarize_unit_canonical_urls(units: Iterable[Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    duplicate_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for unit in units:
        source = _text(_get(unit, "source_project") or _metadata(unit).get("source")) or "unknown"
        url = _canonical_url(unit)
        group = groups.setdefault(
            source,
            {
                "source": source,
                "unit_count": 0,
                "canonical_url_count": 0,
                "missing_canonical_url_count": 0,
                "http_url_count": 0,
                "https_url_count": 0,
            },
        )
        group["unit_count"] += 1
        if not url:
            group["missing_canonical_url_count"] += 1
            continue
        group["canonical_url_count"] += 1
        normalized = _normalize_url(url)
        duplicate_counts[source][normalized] += 1
        scheme = urlsplit(url).scheme.casefold()
        if scheme == "http":
            group["http_url_count"] += 1
        elif scheme == "https":
            group["https_url_count"] += 1

    rows = []
    for source in sorted(groups, key=lambda value: (value.casefold(), value)):
        row = groups[source]
        row["duplicate_canonical_url_count"] = sum(count for count in duplicate_counts[source].values() if count > 1)
        rows.append(row)
    return {"rows": rows, "row_count": len(rows), "unit_count": sum(row["unit_count"] for row in rows)}


def _canonical_url(unit: Any) -> str:
    meta = _metadata(unit)
    for key in _URL_KEYS:
        value = _text(_get(unit, key))
        if value:
            return value
        value = _text(meta.get(key))
        if value:
            return value
    return ""


def _normalize_url(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value.strip()
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path, parsed.query, parsed.fragment))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _metadata(value: object) -> Mapping[str, Any]:
    raw = _get(value, "metadata")
    return raw if isinstance(raw, Mapping) else {}


def _text(value: object) -> str:
    text = "" if value is None else str(getattr(value, "value", value))
    return " ".join(text.split())
