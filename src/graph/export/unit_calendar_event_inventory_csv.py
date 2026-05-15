"""CSV export for calendar-like unit metadata."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_project",
    "source_entity_type",
    "start_date",
    "start_datetime",
    "end_date",
    "end_datetime",
    "due_date",
    "event_date",
    "duration_minutes",
    "attendee_count",
    "location",
    "calendar_id",
]
_START_KEYS = ("start", "start_at", "starts_at")
_END_KEYS = ("end", "end_at", "ends_at")
_DUE_KEYS = ("due", "due_date")
_EVENT_DATE_KEYS = ("event_date",)
_EVENT_KEYS = _START_KEYS + _END_KEYS + _DUE_KEYS + _EVENT_DATE_KEYS + ("location", "attendees", "calendar_id")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_calendar_event_inventory_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write an inventory of units with event-like metadata."""
    unit_list = list(units)
    rows = _event_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "event_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _event_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
        if not any(key in metadata for key in _EVENT_KEYS):
            continue
        start = _first_datetime(metadata, _START_KEYS)
        end = _first_datetime(metadata, _END_KEYS)
        due = _first_datetime(metadata, _DUE_KEYS)
        event_date = _first_datetime(metadata, _EVENT_DATE_KEYS)
        rows.append(
            {
                "unit_id": _field_value(unit.id),
                "title": _inline_text(unit.title),
                "source_project": _field_value(unit.source_project),
                "source_entity_type": _field_value(unit.source_entity_type),
                "start_date": start.date().isoformat() if start else "",
                "start_datetime": start.isoformat() if start else "",
                "end_date": end.date().isoformat() if end else "",
                "end_datetime": end.isoformat() if end else "",
                "due_date": due.date().isoformat() if due else "",
                "event_date": event_date.date().isoformat() if event_date else "",
                "duration_minutes": _duration_minutes(start, end),
                "attendee_count": _attendee_count(metadata.get("attendees")),
                "location": _inline_text(metadata.get("location")),
                "calendar_id": _inline_text(metadata.get("calendar_id")),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["start_date"] or "9999-12-31",
            _sort_key(row["unit_id"]),
        ),
    )


def _first_datetime(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> datetime | None:
    for key in keys:
        parsed = _datetime_value(metadata.get(key))
        if parsed is not None:
            return parsed
    return None


def _datetime_value(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    text = _inline_text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.combine(date.fromisoformat(text), time.min)
        except ValueError:
            return None


def _duration_minutes(start: datetime | None, end: datetime | None) -> int | str:
    if start is None or end is None:
        return ""
    start_utc = _as_utc(start)
    end_utc = _as_utc(end)
    return int((end_utc - start_utc).total_seconds() // 60)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _attendee_count(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len([part for part in re.split(r"[;,]", value) if _inline_text(part)])
    if isinstance(value, list | tuple | set):
        return len([item for item in value if _inline_text(item)])
    return 1 if _inline_text(value) else 0


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
