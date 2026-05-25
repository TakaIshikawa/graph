"""Summarize source terms-of-use review freshness."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

_HOST_KEYS = ("host", "hostname", "domain")
_URL_KEYS = ("url", "source_url", "terms_url")
_REVIEWED_AT_KEYS = ("terms_reviewed_at", "reviewed_at", "last_terms_review_at", "terms_review_date")


def summarize_source_terms_reviews(
    sources: Iterable[Any],
    *,
    reference_date: str | datetime,
    max_age_days: int = 365,
) -> dict[str, Any]:
    """Aggregate source terms review age and stale hosts."""

    if max_age_days < 0:
        raise ValueError("max_age_days must be non-negative")
    reference = _parse_dt(reference_date)
    reviewed = stale = missing = 0
    review_ages: list[int] = []
    stale_hosts: set[str] = set()

    for source in sources:
        metadata = _metadata(source)
        host = _host(_first(source, metadata, _HOST_KEYS), _first(source, metadata, _URL_KEYS))
        reviewed_at = _first(source, metadata, _REVIEWED_AT_KEYS)
        if reviewed_at in (None, ""):
            missing += 1
            continue
        reviewed += 1
        age = (reference - _parse_dt(reviewed_at)).days
        review_ages.append(age)
        if age > max_age_days:
            stale += 1
            if host:
                stale_hosts.add(host)

    return {
        "reviewed_sources": reviewed,
        "stale_review_count": stale,
        "missing_review_count": missing,
        "average_days_since_review": sum(review_ages) / len(review_ages) if review_ages else 0.0,
        "stale_hosts": sorted(stale_hosts),
    }


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


def _parse_dt(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
