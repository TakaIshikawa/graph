"""CSV export for collection reading velocity."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, normalized_key, parse_datetime, render_csv, sort_key, write_csv

_FIELDNAMES = ["collection", "total_units", "started_units", "completed_units", "average_progress_percent", "latest_activity_date", "units_per_week"]
_COLLECTION_KEYS = {"collection", "collections", "collection_id", "collection_name", "project", "list", "folder"}
_UNASSIGNED = "unassigned"


def export_collection_reading_velocity_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = _rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(units: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    groups: dict[str, list[Mapping[str, Any] | object]] = defaultdict(list)
    for unit in units:
        for collection in _collections(unit):
            groups[collection].append(unit)

    rows = []
    for collection, members in groups.items():
        progresses = [_progress(unit) for unit in members]
        dates = [date for unit in members if (date := _activity_date(unit))]
        completed = sum(1 for unit, progress in zip(members, progresses, strict=True) if _status(unit, progress) == "completed")
        started = sum(1 for unit, progress in zip(members, progresses, strict=True) if _status(unit, progress) != "not_started")
        rows.append(
            {
                "collection": collection,
                "total_units": len(members),
                "started_units": started,
                "completed_units": completed,
                "average_progress_percent": f"{sum(progresses) / len(progresses):.1f}" if progresses else "0.0",
                "latest_activity_date": max(dates).date().isoformat() if dates else "",
                "units_per_week": f"{_units_per_week(members, dates):.2f}",
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


def _progress(unit: Mapping[str, Any] | object) -> float:
    data = metadata(unit)
    explicit = _number(data.get("progress") or data.get("progress_percent"))
    if explicit is not None:
        return _clamp(explicit * 100 if 0 <= explicit <= 1 else explicit)
    pages_read = _number(data.get("pages_read") or data.get("current_page"))
    total_pages = _number(data.get("total_pages") or data.get("pages"))
    if pages_read is not None and total_pages and total_pages > 0:
        return _clamp((pages_read / total_pages) * 100)
    status = field_value(data.get("status") or data.get("reading_status")).casefold()
    return 100.0 if status in {"completed", "complete", "done", "finished", "read"} else 0.0


def _status(unit: Mapping[str, Any] | object, progress: float) -> str:
    raw = field_value(metadata(unit).get("status") or metadata(unit).get("reading_status")).casefold()
    if raw in {"completed", "complete", "done", "finished", "read"} or progress >= 100:
        return "completed"
    if raw in {"in_progress", "reading", "started"} or progress > 0:
        return "in_progress"
    return "not_started"


def _activity_date(unit: Mapping[str, Any] | object):
    data = metadata(unit)
    return parse_datetime(
        get(unit, "updated_at")
        or get(unit, "created_at")
        or data.get("last_read_at")
        or data.get("latest_activity_date")
        or data.get("updated_at")
        or data.get("created_at")
        or data.get("date")
    )


def _units_per_week(members: list[Mapping[str, Any] | object], dates: list[Any]) -> float:
    if not dates:
        return 0.0
    span = max(dates) - min(dates)
    if span < timedelta(days=1):
        return float(len(members))
    return len(members) / max(span.days / 7, 1)


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        text = field_value(value).removesuffix("%")
        return float(text) if text else None
    except ValueError:
        return None


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))
