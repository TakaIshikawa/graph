"""Monthly unit growth summary by source."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any


def source_unit_growth_summary(units: Iterable[Mapping[str, Any] | object]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, list[datetime | None]]] = defaultdict(lambda: defaultdict(list))
    for unit in units:
        source = _text(_get(unit, "source_project")) or "unknown"
        timestamp = _unit_timestamp(unit)
        month = timestamp.strftime("%Y-%m") if timestamp else "invalid_timestamp"
        groups[source][month].append(timestamp)

    rows: list[dict[str, Any]] = []
    for source in sorted(groups, key=_sort_key):
        cumulative = 0
        for month in sorted(groups[source], key=lambda value: (value == "invalid_timestamp", value)):
            timestamps = [value for value in groups[source][month] if value is not None]
            count = len(groups[source][month])
            cumulative += count
            rows.append(
                {
                    "source_project": source,
                    "month": month,
                    "unit_count": count,
                    "cumulative_count": cumulative,
                    "first_timestamp": min(timestamps).isoformat() if timestamps else "",
                    "latest_timestamp": max(timestamps).isoformat() if timestamps else "",
                }
            )
    return rows


def _unit_timestamp(unit: Mapping[str, Any] | object) -> datetime | None:
    for key in ("created_at", "ingested_at", "updated_at"):
        parsed = _parse_datetime(_get(unit, key))
        if parsed:
            return parsed
    metadata = _metadata(unit)
    for key in ("created_at", "created", "ingested_at", "ingested", "updated_at", "updated"):
        parsed = _parse_datetime(metadata.get(key))
        if parsed:
            return parsed
    return None


def _metadata(unit: Mapping[str, Any] | object) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = _text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _get(value: Mapping[str, Any] | object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _text(value: object) -> str:
    return "" if value is None else str(getattr(value, "value", value)).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
