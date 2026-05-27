"""Content freshness summary by source and entity type."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any


def unit_content_freshness_summary(
    units: Iterable[Mapping[str, Any] | object],
    *,
    reference_date: date | datetime | str | None = None,
    stale_after_days: int | None = None,
) -> list[dict[str, Any]]:
    ref_date = _parse_date(reference_date)
    grouped: dict[tuple[str, str], list[Mapping[str, Any] | object]] = defaultdict(list)
    for unit in units:
        grouped[(_text(_get(unit, "source_project")) or "unknown", _text(_get(unit, "source_entity_type")) or "unknown")].append(unit)

    rows: list[dict[str, Any]] = []
    for (source, entity_type), source_units in sorted(grouped.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))):
        created_dates = [_unit_date(unit, "created_at", ("created_at", "created")) for unit in source_units]
        updated_dates = [_unit_date(unit, "updated_at", ("updated_at", "updated")) for unit in source_units]
        valid_updated = [value for value in updated_dates if value is not None]
        rows.append(
            {
                "source_project": source,
                "source_entity_type": entity_type,
                "unit_count": len(source_units),
                "missing_created_at_count": sum(1 for value in created_dates if value is None),
                "missing_updated_at_count": sum(1 for value in updated_dates if value is None),
                "future_updated_at_count": sum(1 for value in valid_updated if ref_date is not None and value > ref_date),
                "stale_unit_count": _stale_count(valid_updated, ref_date, stale_after_days),
                "latest_updated_at": max(valid_updated).isoformat() if valid_updated else "",
            }
        )
    return rows


def _stale_count(values: list[date], reference_date: date | None, stale_after_days: int | None) -> int:
    if reference_date is None or stale_after_days is None:
        return 0
    return sum(1 for value in values if (reference_date - value).days > stale_after_days)


def _unit_date(unit: Mapping[str, Any] | object, attr: str, metadata_keys: tuple[str, ...]) -> date | None:
    parsed = _parse_date(_get(unit, attr))
    if parsed:
        return parsed
    metadata = _metadata(unit)
    for key in metadata_keys:
        parsed = _parse_date(metadata.get(key))
        if parsed:
            return parsed
    return None


def _metadata(unit: Mapping[str, Any] | object) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _parse_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _get(value: Mapping[str, Any] | object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _text(value: object) -> str:
    return "" if value is None else str(getattr(value, "value", value)).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
