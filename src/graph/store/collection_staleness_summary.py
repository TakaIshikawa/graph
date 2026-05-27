"""Summarize collection recency and staleness buckets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any

_DATE_KEYS = ("updated_at", "modified_at", "last_seen_at")
_ID_KEYS = ("id", "collection_id")
_MEMBER_KEYS = ("member_ids", "members", "unit_ids")


def collection_staleness_summary(
    collections: Iterable[Any], *, reference_date: Any = None, stale_after_days: int = 90
) -> list[dict[str, Any]]:
    reference = _parse_date(reference_date) if reference_date is not None else datetime.now(timezone.utc).date()
    rows = []
    for collection in collections:
        metadata = _metadata(collection)
        collection_id = _string(_first(collection, metadata, _ID_KEYS))
        updated = _parse_date(_first(collection, metadata, _DATE_KEYS))
        age_days = (reference - updated).days if updated is not None else None
        bucket = _bucket(age_days, stale_after_days)
        rows.append(
            {
                "collection_id": collection_id,
                "age_days": age_days,
                "staleness_bucket": bucket,
                "is_stale": bucket == "stale",
                "member_count": _member_count(_first(collection, metadata, _MEMBER_KEYS)),
                "sample_collection_ids": [collection_id] if collection_id else [],
            }
        )
    return sorted(rows, key=lambda row: (row["staleness_bucket"], row["collection_id"] or ""))


def _bucket(age_days: int | None, stale_after_days: int) -> str:
    if age_days is None:
        return "unknown"
    if age_days > stale_after_days:
        return "stale"
    if age_days > stale_after_days // 2:
        return "aging"
    return "fresh"


def _member_count(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.date()


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
