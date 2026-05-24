"""Compare collection tag usage between timestamped snapshots."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any


def collection_tag_transition_summary(
    units: Iterable[Any],
    *,
    before_at: str | datetime,
    after_at: str | datetime,
) -> list[dict[str, Any]]:
    before = _parse_dt(before_at)
    after = _parse_dt(after_at)
    if before > after:
        raise ValueError("before_at must be on or before after_at.")
    prior: dict[str, set[str]] = defaultdict(set)
    current: dict[str, set[str]] = defaultdict(set)
    collections: set[str] = set()
    for unit in units:
        timestamp = _parse_dt(_timestamp(unit))
        names = _collections(unit)
        collections.update(names)
        tags = {str(tag) for tag in (_get(unit, "tags") or [])}
        for name in names:
            if timestamp <= before:
                prior[name].update(tags)
            if before < timestamp <= after:
                current[name].update(tags)
    rows = []
    for name in sorted(collections):
        before_tags = prior.get(name, set())
        after_tags = current.get(name, set())
        added = sorted(after_tags - before_tags)
        removed = sorted(before_tags - after_tags)
        retained = sorted(before_tags & after_tags)
        rows.append(
            {
                "collection": name,
                "added_count": len(added),
                "removed_count": len(removed),
                "retained_count": len(retained),
                "net_change": len(after_tags) - len(before_tags),
                "added_tags": added,
                "removed_tags": removed,
                "retained_tags": retained,
            }
        )
    return rows


def _collections(unit: Any) -> list[str]:
    metadata = _metadata(unit)
    value = metadata.get("collections", metadata.get("collection"))
    if isinstance(value, list):
        return sorted(str(item) for item in value if str(item))
    if value not in (None, ""):
        return [str(value)]
    return []


def _timestamp(unit: Any) -> Any:
    metadata = _metadata(unit)
    return (
        metadata.get("snapshot_at")
        or metadata.get("collection_snapshot_at")
        or _get(unit, "updated_at")
        or _get(unit, "created_at")
    )


def _parse_dt(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)
