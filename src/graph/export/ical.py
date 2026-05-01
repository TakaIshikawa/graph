"""iCalendar export helpers for dated graph units."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from graph.types.models import KnowledgeUnit

DATE_METADATA_KEYS = (
    "due_date",
    "start_date",
    "event_date",
    "scheduled_at",
    "published_at",
)


def export_units_to_ics(
    units: Iterable[KnowledgeUnit],
    path: str | Path,
    *,
    units_scanned: int | None = None,
) -> dict:
    """Write dated units as VEVENT entries to an iCalendar file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    events: list[list[str]] = []
    scanned = 0
    for unit in units:
        scanned += 1
        event_date = unit_event_datetime(unit)
        if event_date is None:
            continue
        events.append(_unit_event_lines(unit, event_date, now))

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Graph//Dated Units Export//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]
    for event in events:
        lines.extend(event)
    lines.append("END:VCALENDAR")

    output_path.write_text(_ical_text(lines), encoding="utf-8")
    return {
        "path": str(output_path),
        "units_scanned": units_scanned if units_scanned is not None else scanned,
        "events_exported": len(events),
    }


def unit_event_datetime(unit: KnowledgeUnit) -> datetime | date | None:
    """Return the first recognized date metadata value for a unit."""
    for key in DATE_METADATA_KEYS:
        if key not in unit.metadata:
            continue
        parsed = _parse_date_value(unit.metadata.get(key))
        if parsed is not None:
            return parsed
    return None


def _unit_event_lines(
    unit: KnowledgeUnit,
    event_date: datetime | date,
    generated_at: datetime,
) -> list[str]:
    description_parts = [
        f"Source: {unit.source_project}/{unit.source_entity_type}",
        f"Graph ID: {unit.id}",
    ]
    if unit.tags:
        description_parts.append(f"Tags: {', '.join(unit.tags)}")
    content = _short_text(unit.content, max_length=500)
    if content:
        description_parts.extend(["", content])

    lines = [
        "BEGIN:VEVENT",
        f"UID:{_escape_text(f'{unit.id}@graph.local')}",
        f"DTSTAMP:{_format_datetime(generated_at)}",
        _date_property("DTSTART", event_date),
        f"SUMMARY:{_escape_text(unit.title or 'Untitled graph unit')}",
        f"DESCRIPTION:{_escape_text(chr(10).join(description_parts))}",
        f"X-GRAPH-UNIT-ID:{_escape_text(unit.id)}",
        f"X-GRAPH-SOURCE-PROJECT:{_escape_text(str(unit.source_project))}",
        f"X-GRAPH-CONTENT-TYPE:{_escape_text(str(unit.content_type))}",
    ]
    if unit.tags:
        lines.append(f"CATEGORIES:{_categories_value(unit.tags)}")
    lines.append("END:VEVENT")
    return lines


def _parse_date_value(value: object) -> datetime | date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _aware_utc(value)
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").date()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None
    try:
        if text.endswith("Z") and "-" not in text:
            return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        if "T" in text and "-" not in text:
            return datetime.strptime(text, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return _aware_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        return None


def _date_property(name: str, value: datetime | date) -> str:
    if isinstance(value, datetime):
        return f"{name}:{_format_datetime(value)}"
    return f"{name};VALUE=DATE:{value.strftime('%Y%m%d')}"


def _format_datetime(value: datetime) -> str:
    return _aware_utc(value).strftime("%Y%m%dT%H%M%SZ")


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _short_text(value: str, *, max_length: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _escape_text(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\n")
    )


def _categories_value(tags: list[str]) -> str:
    return ",".join(_escape_text(tag) for tag in tags)


def _ical_text(lines: list[str]) -> str:
    folded = []
    for line in lines:
        folded.extend(_fold_line(line))
    return "\r\n".join(folded) + "\r\n"


def _fold_line(line: str) -> list[str]:
    if len(line) <= 75:
        return [line]
    chunks = [line[:75]]
    remainder = line[75:]
    while remainder:
        chunks.append(" " + remainder[:74])
        remainder = remainder[74:]
    return chunks
