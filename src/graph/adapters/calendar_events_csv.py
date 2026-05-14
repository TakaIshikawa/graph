"""Adapter for generic calendar event CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class CalendarEventsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "calendar_events_csv"

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
        units: list[KnowledgeUnit] = []
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
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.updated_at, unit.source_id)))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Subject", "Summary", "Event", "Name")
        description = first(row, "Description", "Notes", "Body", "Details")
        location = first(row, "Location", "Where", "Venue")
        start = parse_datetime(first(row, "Start", "Start Date", "Start Time", "Starts", "Begin", "Begin Time"))
        end = parse_datetime(first(row, "End", "End Date", "End Time", "Ends", "Finish", "Finish Time"))
        updated = parse_datetime(first(row, "Updated", "Updated At", "Modified", "Last Modified"))
        if not (title or description or location or start or end):
            return None
        event_at = updated or start or end or datetime.now(timezone.utc)
        organizer = first(row, "Organizer", "Organiser", "Created By", "Owner")
        attendees = split_values(first(row, "Attendees", "Guests", "Participants", "Invitees"))
        url = first(row, "URL", "Url", "Link", "Meet Link", "Conference URL")
        all_day = self._parse_bool(first(row, "All Day", "All-day", "AllDay")) or self._looks_all_day(row)
        metadata = clean_metadata(
            {
                "title": title,
                "description": description,
                "location": location,
                "start_at": start.isoformat() if start else None,
                "end_at": end.isoformat() if end else None,
                "updated_at": updated.isoformat() if updated else None,
                "attendees": attendees,
                "organizer": organizer,
                "all_day": all_day,
                "url": url,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.CALENDAR_EVENTS_CSV,
            source_id=digest_source_id("calendar_events_csv", first(row, "ID", "Event ID", "Uid", "UID") or title, start, end, location, index),
            source_entity_type="calendar_event",
            title=title or "Untitled calendar event",
            content=self._content(title or "Untitled calendar event", description, location, start, end, attendees, organizer, url),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["calendar", "calendar_event"],
            created_at=start or event_at,
            updated_at=event_at,
        )

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

    def _looks_all_day(self, row: dict[str, Any]) -> bool | None:
        start_text = first(row, "Start", "Start Date", "Starts")
        end_text = first(row, "End", "End Date", "Ends")
        if start_text and len(start_text.strip()) <= 10 and end_text and len(end_text.strip()) <= 10:
            return True
        return None

    def _content(
        self,
        title: str,
        description: str,
        location: str,
        start: datetime | None,
        end: datetime | None,
        attendees: list[str],
        organizer: str,
        url: str,
    ) -> str:
        parts = [title]
        if description:
            parts.append(description)
        if location:
            parts.append(f"Location: {location}")
        if start:
            parts.append(f"Start: {start.isoformat()}")
        if end:
            parts.append(f"End: {end.isoformat()}")
        if organizer:
            parts.append(f"Organizer: {organizer}")
        if attendees:
            parts.append(f"Attendees: {', '.join(attendees)}")
        if url:
            parts.append(f"URL: {url}")
        return "\n".join(parts)
