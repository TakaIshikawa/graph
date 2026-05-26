"""CSV export for collection member recency."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, parse_datetime, render_csv, sort_key, write_csv

_FIELDNAMES = ["collection", "unit_count", "earliest_updated_at", "latest_updated_at", "stale_unit_count", "missing_timestamp_count"]
_COLLECTION_KEYS = ("collection", "collections", "folder", "project", "notebook")
_TIME_KEYS = ("updated_at", "modified_at", "last_updated", "timestamp")


def export_collection_member_recency_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    *,
    stale_before: datetime | str | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    stale_cutoff = parse_datetime(stale_before)
    buckets: dict[str, list[Mapping[str, Any] | object]] = defaultdict(list)
    for unit in unit_list:
        for collection in _collections(unit):
            buckets[collection].append(unit)
    rows = [_row(collection, grouped, stale_cutoff) for collection, grouped in sorted(buckets.items(), key=lambda item: sort_key(item[0]))]
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(collection: str, units: list[Mapping[str, Any] | object], stale_before: datetime | None) -> dict[str, str | int]:
    parsed = [_updated_at(unit) for unit in units]
    valid = [value for value in parsed if value]
    return {
        "collection": collection,
        "unit_count": len(units),
        "earliest_updated_at": min(valid).isoformat() if valid else "",
        "latest_updated_at": max(valid).isoformat() if valid else "",
        "stale_unit_count": sum(1 for value in valid if stale_before and value < stale_before),
        "missing_timestamp_count": sum(1 for value in parsed if not value),
    }


def _collections(unit: Mapping[str, Any] | object) -> list[str]:
    meta = metadata(unit)
    found = [field_value(value) for key in _COLLECTION_KEYS for value in flatten_values(get(unit, key) or meta.get(key)) if field_value(value)]
    return sorted(set(found), key=sort_key) or ["unassigned"]


def _updated_at(unit: Mapping[str, Any] | object) -> datetime | None:
    meta = metadata(unit)
    for key in _TIME_KEYS:
        parsed = parse_datetime(get(unit, key) or meta.get(key))
        if parsed:
            return parsed
    return None
