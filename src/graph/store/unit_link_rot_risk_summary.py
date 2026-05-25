"""Summarize link rot risk from unit link-check metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

_LINK_LIST_KEYS = ("links", "urls", "external_links", "link_checks")
_STATUS_KEYS = ("status", "status_code", "http_status", "response_status")
_ARCHIVED_KEYS = ("archived", "is_archived", "archive_url")
_CHECKED_AT_KEYS = ("checked_at", "last_checked_at", "fetched_at")


def summarize_unit_link_rot_risk(
    units: Iterable[Any],
    *,
    reference_date: str | datetime | None = None,
    stale_after_days: int = 30,
) -> dict[str, Any]:
    """Bucket units by link-check failure, archive, and staleness signals."""

    if stale_after_days < 0:
        raise ValueError("stale_after_days must be non-negative")
    reference = _parse_dt(reference_date) if reference_date is not None else None

    buckets: Counter[str] = Counter()
    total_units = urls = failing_urls = archived_urls = stale_checks = 0
    for unit in units:
        total_units += 1
        links = _links(unit)
        if not links:
            buckets["no_links"] += 1
            continue

        unit_failing = unit_archived = unit_stale = False
        for link in links:
            urls += 1
            if _is_failing(link):
                failing_urls += 1
                unit_failing = True
            if _is_archived(link):
                archived_urls += 1
                unit_archived = True
            if _is_stale(link, reference, stale_after_days):
                stale_checks += 1
                unit_stale = True

        if unit_failing:
            buckets["failing"] += 1
        elif unit_archived:
            buckets["archived"] += 1
        elif unit_stale:
            buckets["stale"] += 1
        else:
            buckets["clean"] += 1

    return {
        "total_units": total_units,
        "urls": urls,
        "failing_urls": failing_urls,
        "archived_urls": archived_urls,
        "stale_checks": stale_checks,
        "risk_buckets": {key: buckets.get(key, 0) for key in ("no_links", "clean", "archived", "stale", "failing")},
    }


def _links(unit: Any) -> list[Mapping[str, Any]]:
    metadata = _metadata(unit)
    for key in _LINK_LIST_KEYS:
        value = metadata.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _is_failing(link: Mapping[str, Any]) -> bool:
    status = _first(link, _STATUS_KEYS)
    if isinstance(status, str) and status.lower() in {"failed", "failure", "error", "timeout", "broken"}:
        return True
    try:
        return int(status) >= 400
    except (TypeError, ValueError):
        return False


def _is_archived(link: Mapping[str, Any]) -> bool:
    return any(bool(link.get(key)) for key in _ARCHIVED_KEYS)


def _is_stale(link: Mapping[str, Any], reference: datetime | None, stale_after_days: int) -> bool:
    if link.get("stale") is True or link.get("stale_check") is True:
        return True
    if reference is None:
        return False
    checked_at = _first(link, _CHECKED_AT_KEYS)
    if checked_at in (None, ""):
        return False
    age = reference - _parse_dt(checked_at)
    return age.days > stale_after_days


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _first(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_dt(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
