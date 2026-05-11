"""Deterministic iCalendar timeline export for dated graph units."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Iterable

from graph.types.models import KnowledgeUnit

DATE_METADATA_KEYS = (
    "date",
    "due_date",
    "start_date",
    "event_date",
    "scheduled_at",
    "published_at",
)


def export_units_to_ical_timeline(
    units: Iterable[KnowledgeUnit],
    *,
    prodid: str = "-//Graph//Timeline Export//EN",
) -> str:
    """Return dated units as a deterministic VCALENDAR string."""
    events: list[tuple[str, list[str]]] = []
    for unit in units:
        event_date = unit_event_datetime(unit)
        if event_date is None:
            continue
        events.append((_event_sort_key(unit, event_date), _event_lines(unit, event_date)))

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:{_escape_text(prodid)}",
        "CALSCALE:GREGORIAN",
    ]
    for _, event_lines in sorted(events, key=lambda item: item[0]):
        lines.extend(event_lines)
    lines.append("END:VCALENDAR")
    return _ical_text(lines)


def unit_event_datetime(unit: KnowledgeUnit) -> datetime | date | None:
    """Return the first usable timeline date on a unit."""
    for key in DATE_METADATA_KEYS:
        if key not in unit.metadata:
            continue
        parsed = _parse_date_value(unit.metadata.get(key))
        if parsed is not None:
            return parsed
    return None


def _event_lines(unit: KnowledgeUnit, event_date: datetime | date) -> list[str]:
    lines = [
        "BEGIN:VEVENT",
        f"UID:{_escape_text(_uid(unit))}",
        "DTSTAMP:19700101T000000Z",
        _date_property("DTSTART", event_date),
        f"SUMMARY:{_escape_text(unit.title or 'Untitled graph unit')}",
    ]
    description = _description(unit)
    if description:
        lines.append(f"DESCRIPTION:{_escape_text(description)}")
    lines.append("END:VEVENT")
    return lines


def _description(unit: KnowledgeUnit) -> str:
    parts = []
    if unit.content:
        parts.append(str(unit.content))
    if unit.tags:
        parts.append(f"Tags: {', '.join(sorted(unit.tags))}")
    return "\n".join(parts)


def _uid(unit: KnowledgeUnit) -> str:
    if unit.id:
        return f"{unit.id}@graph.local"
    return f"{unit.source_project}:{unit.source_id}@graph.local"


def _event_sort_key(unit: KnowledgeUnit, event_date: datetime | date) -> str:
    return f"{_date_sort_text(event_date)}\0{unit.id}\0{unit.source_id}"


def _date_sort_text(value: datetime | date) -> str:
    if isinstance(value, datetime):
        return _aware_utc(value).isoformat()
    return value.isoformat()


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
        try:
            return datetime.strptime(text, "%Y%m%d").date()
        except ValueError:
            return None
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
        return f"{name}:{_aware_utc(value).strftime('%Y%m%dT%H%M%SZ')}"
    return f"{name};VALUE=DATE:{value.strftime('%Y%m%d')}"


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _escape_text(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\n")
    )


def _ical_text(lines: list[str]) -> str:
    return "\r\n".join(_folded_lines(lines)) + "\r\n"


def _folded_lines(lines: list[str]) -> list[str]:
    folded: list[str] = []
    for line in lines:
        if len(line) <= 75:
            folded.append(line)
            continue
        folded.append(line[:75])
        remainder = line[75:]
        while remainder:
            folded.append(" " + remainder[:74])
            remainder = remainder[74:]
    return folded
