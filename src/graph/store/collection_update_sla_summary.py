"""Summarize collection update SLA freshness metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

_SLA_KEYS = ("sla_days", "update_sla_days", "freshness_sla_days")
_LAST_UPDATED_KEYS = ("last_updated_at", "updated_at", "last_refreshed_at")
_OWNER_KEYS = ("owner", "owner_id", "team")


def summarize_collection_update_slas(
    collections: Iterable[Any],
    *,
    reference_date: str | datetime,
) -> dict[str, Any]:
    """Aggregate collection freshness SLA status using a fixed reference date."""

    reference = _parse_dt(reference_date)
    total = configured = breached = missing = 0
    breach_days: list[int] = []
    breach_by_owner: Counter[str | None] = Counter()

    for collection in collections:
        total += 1
        metadata = _metadata(collection)
        sla_days = _positive_int(_first(collection, metadata, _SLA_KEYS))
        if sla_days is None:
            continue
        configured += 1
        owner = _string(_first(collection, metadata, _OWNER_KEYS))
        last_updated = _first(collection, metadata, _LAST_UPDATED_KEYS)
        if last_updated in (None, ""):
            missing += 1
            breached += 1
            breach_by_owner[owner] += 1
            continue
        days_since_update = (reference - _parse_dt(last_updated)).days
        if days_since_update > sla_days:
            breached += 1
            breach_days.append(days_since_update - sla_days)
            breach_by_owner[owner] += 1

    return {
        "total_collections": total,
        "configured_collections": configured,
        "breached_collections": breached,
        "missing_last_updated_count": missing,
        "average_breach_days": sum(breach_days) / len(breach_days) if breach_days else 0.0,
        "breach_by_owner": [
            {"owner": owner, "count": count}
            for owner, count in sorted(breach_by_owner.items(), key=lambda item: item[0] or "")
        ],
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


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_dt(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
