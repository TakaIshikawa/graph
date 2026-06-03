"""CSV export for collection staleness risk."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from graph.export._report_csv import field_value, get, metadata, parse_datetime, render_csv, write_csv

_FIELDS = ["collection", "unit_count", "oldest_updated", "newest_updated", "median_age_days", "stale_unit_count", "risk_level"]


def export_collection_staleness_risk_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    *,
    stale_after_days: int = 90,
    now: datetime | None = None,
) -> str | dict[str, Any]:
    if stale_after_days < 0:
        raise ValueError("stale_after_days must be non-negative")
    now = _ensure_utc(now or datetime.now(timezone.utc))
    groups: dict[str, list[datetime]] = defaultdict(list)
    unit_list = list(units)
    for unit in unit_list:
        updated = _updated_at(unit)
        if updated is None:
            continue
        collections = _collections(unit) or ["unassigned"]
        for collection in collections:
            groups[collection].append(updated)
    rows = [_row(collection, values, stale_after_days, now) for collection, values in sorted(groups.items())]
    text = render_csv(rows, _FIELDS)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(collection: str, updated_values: list[datetime], stale_after_days: int, now: datetime) -> dict[str, Any]:
    ages = [max((now - value).total_seconds() / 86400, 0) for value in updated_values]
    stale_count = sum(1 for age in ages if age > stale_after_days)
    stale_ratio = stale_count / len(updated_values)
    risk = "high" if stale_ratio >= 0.75 else "medium" if stale_count else "low"
    return {
        "collection": collection,
        "unit_count": len(updated_values),
        "oldest_updated": min(updated_values).isoformat(),
        "newest_updated": max(updated_values).isoformat(),
        "median_age_days": round(median(ages), 2),
        "stale_unit_count": stale_count,
        "risk_level": risk,
    }


def _collections(unit: Mapping[str, Any] | object) -> list[str]:
    raw = get(unit, "collections")
    if raw is None:
        raw = metadata(unit).get("collections") or metadata(unit).get("collection")
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, list | tuple | set):
        return [field_value(item) for item in raw if field_value(item)]
    text = field_value(raw)
    return [text] if text else []


def _updated_at(unit: Mapping[str, Any] | object) -> datetime | None:
    for value in (get(unit, "updated_at"), metadata(unit).get("updated_at"), get(unit, "created_at")):
        parsed = parse_datetime(value)
        if parsed is not None:
            return _ensure_utc(parsed)
    return None


def _ensure_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
