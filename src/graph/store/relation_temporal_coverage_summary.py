"""Summarize relation timestamp coverage by relation type."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any

_RELATION_KEYS = ("relation_type", "relation", "type", "predicate")
_DATE_KEYS = ("created_at", "updated_at", "observed_at", "start_date", "end_date")
_ID_KEYS = ("id", "relation_id", "edge_id")


def relation_temporal_coverage_summary(relations: Iterable[Any]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for relation in relations:
        metadata = _metadata(relation)
        relation_type = _string(_first(relation, metadata, _RELATION_KEYS)) or "unknown"
        group = groups.setdefault(
            relation_type,
            {
                "relation_type": relation_type,
                "total_count": 0,
                "dated_count": 0,
                "dates": [],
                "sample_relation_ids": [],
            },
        )
        group["total_count"] += 1
        parsed_dates = [_parse_date(_first(relation, metadata, (key,))) for key in _DATE_KEYS]
        parsed_dates = [value for value in parsed_dates if value is not None]
        if parsed_dates:
            group["dated_count"] += 1
            group["dates"].extend(parsed_dates)
        relation_id = _string(_first(relation, metadata, _ID_KEYS))
        if relation_id and len(group["sample_relation_ids"]) < 3:
            group["sample_relation_ids"].append(relation_id)

    rows = []
    for group in groups.values():
        total = group["total_count"]
        dates = group["dates"]
        rows.append(
            {
                "relation_type": group["relation_type"],
                "total_count": total,
                "dated_count": group["dated_count"],
                "missing_date_count": total - group["dated_count"],
                "coverage_share": round(group["dated_count"] / total, 4) if total else 0,
                "earliest_date": min(dates).isoformat() if dates else None,
                "latest_date": max(dates).isoformat() if dates else None,
                "sample_relation_ids": group["sample_relation_ids"],
            }
        )
    return sorted(rows, key=lambda row: (-row["total_count"], row["relation_type"]))


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
