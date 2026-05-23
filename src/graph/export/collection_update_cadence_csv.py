"""CSV export for collection update cadence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, normalized_key, parse_datetime, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["collection", "update_count", "first_seen", "last_seen", "median_gap_days", "stale_after_days", "is_stale"]
_COLLECTION_KEYS = {"collection", "collections", "collection_id", "collection_name", "project", "list", "folder"}
_DATE_KEYS = ("updated_at", "created_at", "ingested_at", "date", "timestamp", "published_at", "modified_at", "last_seen")
_UNASSIGNED = "unassigned"


def export_collection_update_cadence_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    *,
    stale_after_days: int | None = None,
    reference_date: datetime | str | None = None,
) -> str | dict[str, Any]:
    """Return or write collection cadence summaries from unit date metadata."""
    unit_list = list(units)
    rows = _cadence_rows(unit_list, stale_after_days, reference_date)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _cadence_rows(units: list[Mapping[str, Any] | object], stale_after_days: int | None, reference_date: datetime | str | None) -> list[dict[str, str | int]]:
    groups: dict[str, list[datetime]] = defaultdict(list)
    for unit in units:
        date = _unit_date(unit)
        for collection in _collections(unit):
            if date:
                groups[collection].append(date)
            else:
                groups.setdefault(collection, [])
    ref = parse_datetime(reference_date) or datetime.now(timezone.utc)
    rows: list[dict[str, str | int]] = []
    for collection, dates in groups.items():
        ordered = sorted(dates)
        gaps = [(right - left).total_seconds() / 86400 for left, right in zip(ordered, ordered[1:])]
        last_seen = ordered[-1] if ordered else None
        rows.append(
            {
                "collection": collection,
                "update_count": len(ordered),
                "first_seen": ordered[0].isoformat() if ordered else "",
                "last_seen": last_seen.isoformat() if last_seen else "",
                "median_gap_days": f"{median(gaps):.2f}" if gaps else "",
                "stale_after_days": "" if stale_after_days is None else stale_after_days,
                "is_stale": _stale_text(last_seen, stale_after_days, ref),
            }
        )
    return sorted(rows, key=lambda row: sort_key(row["collection"]))


def _collections(unit: Mapping[str, Any] | object) -> list[str]:
    values: set[str] = set()
    for key in _COLLECTION_KEYS:
        text = field_value(get(unit, key))
        if text:
            values.add(text)
    for key, value in metadata(unit).items():
        if normalized_key(key) in _COLLECTION_KEYS:
            values.update(field_value(item) for item in flatten_values(value) if field_value(item))
    return sorted(values, key=sort_key) or [_UNASSIGNED]


def _unit_date(unit: Mapping[str, Any] | object) -> datetime | None:
    for key in _DATE_KEYS:
        parsed = parse_datetime(get(unit, key))
        if parsed:
            return parsed
    for key in _DATE_KEYS:
        parsed = parse_datetime(metadata(unit).get(key))
        if parsed:
            return parsed
    return None


def _stale_text(last_seen: datetime | None, stale_after_days: int | None, reference_date: datetime) -> str:
    if last_seen is None or stale_after_days is None:
        return "unknown"
    return "true" if (reference_date - last_seen).days > stale_after_days else "false"
