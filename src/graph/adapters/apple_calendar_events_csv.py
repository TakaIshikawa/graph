"""Adapter for Apple Calendar event CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AppleCalendarEventsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "apple_calendar_events_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["calendar_event"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else set(self.entity_types)
        if "calendar_event" not in allowed:
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        units: dict[str, KnowledgeUnit] = {}
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units[unit.source_id] = unit

        result.units.extend(sorted(units.values(), key=lambda unit: (unit.updated_at, unit.source_id)))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Summary", "Subject", "Event", "Name")
        calendar = first(row, "Calendar", "Calendar name", "Calendar Name", "Calendar title")
        start_text = first(row, "Start", "Start date", "Start Date", "Starts", "Start time", "Start Time")
        end_text = first(row, "End", "End date", "End Date", "Ends", "End time", "End Time")
        start = parse_datetime(start_text)
        end = parse_datetime(end_text)
        location = first(row, "Location", "Where", "Venue")
        url = first(row, "URL", "Url", "Link", "Event URL", "Conference URL")
        notes = first(row, "Notes", "Description", "Body", "Details")
        attendees = split_values(first(row, "Attendees", "Invitees", "Guests", "Participants"))
        organizer = first(row, "Organizer", "Organiser", "Owner", "Created By")
        recurrence = first(row, "Recurrence", "Repeat", "Repeats", "RRULE", "Rule")
        all_day_value = self._parse_bool(first(row, "All Day", "All-day", "AllDay", "All day event"))
        all_day = all_day_value if all_day_value is not None else self._looks_all_day(start_text, end_text)
        updated = parse_datetime(first(row, "Updated", "Updated at", "Modified", "Last Modified"))
        event_id = first(row, "ID", "Event ID", "UID", "Uid")
        if not any([title, calendar, start_text, end_text, location, url, notes, attendees, organizer, recurrence]):
            return None

        event_at = updated or start or end or datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "title": title,
                "calendar": calendar,
                "start_at": start.isoformat() if start else start_text,
                "end_at": end.isoformat() if end else end_text,
                "all_day": all_day,
                "location": location,
                "url": url,
                "source_url": url,
                "notes": notes,
                "attendees": attendees,
                "organizer": organizer,
                "recurrence": recurrence,
                "source_file": source_file,
                "source_row": index + 2,
            }
        )
        return KnowledgeUnit(
            source_project="apple_calendar_events_csv",
            source_id=self._source_id(event_id, title, calendar, start_text, end_text, location, index),
            source_entity_type="calendar_event",
            title=title or "Untitled calendar event",
            content=self._content(title or "Untitled calendar event", metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["apple_calendar", "calendar_event", calendar] if tag)),
            created_at=start or event_at,
            updated_at=event_at,
        )

    def _source_id(self, event_id: str, title: str, calendar: str, start: str, end: str, location: str, index: int) -> str:
        if event_id:
            return f"apple_calendar_events_csv:{event_id}"
        return digest_source_id("apple_calendar_events_csv", title, calendar, start, end, location, index)

    def _parse_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if value in (None, ""):
            return None
        text = str(value).strip().casefold()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
        return None

    def _looks_all_day(self, start_text: str, end_text: str) -> bool | None:
        if start_text and len(start_text.strip()) <= 10 and end_text and len(end_text.strip()) <= 10:
            return True
        return None

    def _content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for label, key in (
            ("Calendar", "calendar"),
            ("Start", "start_at"),
            ("End", "end_at"),
            ("Location", "location"),
            ("Organizer", "organizer"),
            ("Attendees", "attendees"),
            ("Recurrence", "recurrence"),
            ("URL", "url"),
            ("Notes", "notes"),
        ):
            if key not in metadata:
                continue
            value = metadata[key]
            if isinstance(value, list):
                value = ", ".join(value)
            parts.append(f"{label}: {value}")
        return "\n".join(parts)
