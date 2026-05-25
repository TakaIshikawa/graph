"""Summarize source sitemap discovery coverage by host."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

_HOST_KEYS = ("host", "hostname", "domain")
_URL_KEYS = ("url", "source_url", "canonical_url", "sitemap_url")
_DISCOVERED_KEYS = (
    "discovered_url_count",
    "sitemap_discovered_url_count",
    "sitemap_url_count",
    "url_count",
)
_INGESTED_KEYS = ("ingested_url_count", "sitemap_ingested_url_count", "indexed_url_count")
_MISSING_KEYS = ("missing_sitemap", "sitemap_missing", "no_sitemap")
_HAS_SITEMAP_KEYS = ("has_sitemap", "sitemap_found", "sitemap_discovered")


def summarize_source_sitemap_coverage(
    sources: Iterable[Any],
    *,
    low_coverage_threshold: float = 0.8,
) -> dict[str, Any]:
    """Aggregate sitemap discovery and ingestion metadata by host."""

    if low_coverage_threshold < 0:
        raise ValueError("low_coverage_threshold must be non-negative")

    hosts: dict[str, dict[str, Any]] = {}
    for source in sources:
        metadata = _metadata(source)
        host = _host(_first(source, metadata, _HOST_KEYS), _first(source, metadata, _URL_KEYS))
        if not host:
            continue
        row = hosts.setdefault(
            host,
            {
                "host": host,
                "discovered_url_count": 0,
                "ingested_url_count": 0,
                "coverage_ratio": 0.0,
                "missing_sitemap_count": 0,
            },
        )
        row["discovered_url_count"] += _non_negative_int(_first(source, metadata, _DISCOVERED_KEYS))
        row["ingested_url_count"] += _non_negative_int(_first(source, metadata, _INGESTED_KEYS))
        if _missing_sitemap(source, metadata):
            row["missing_sitemap_count"] += 1

    rows = []
    low_coverage_hosts = []
    for host in sorted(hosts):
        row = hosts[host]
        discovered = row["discovered_url_count"]
        row["coverage_ratio"] = row["ingested_url_count"] / discovered if discovered else 0.0
        rows.append(row)
        if discovered and row["coverage_ratio"] < low_coverage_threshold:
            low_coverage_hosts.append(host)

    return {"hosts": rows, "low_coverage_hosts": low_coverage_hosts}


def _missing_sitemap(item: Any, metadata: Mapping[str, Any]) -> bool:
    marker = _first(item, metadata, _MISSING_KEYS)
    if marker is not None:
        return bool(marker)
    has_sitemap = _first(item, metadata, _HAS_SITEMAP_KEYS)
    return has_sitemap is False


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _first(item: Any, metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _get(item, key)
        if value not in (None, ""):
            return value
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _host(host_value: Any, url_value: Any) -> str | None:
    if host_value not in (None, ""):
        return str(host_value).strip().lower()
    if url_value in (None, ""):
        return None
    parsed = urlsplit(str(url_value) if "://" in str(url_value) else f"https://{url_value}")
    return parsed.hostname.lower() if parsed.hostname else None


def _non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)
