"""Adapter for Meetup RSVP CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class MeetupRsvpsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "meetup_rsvps_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["rsvp"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "rsvp" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        event_id = first(row, "event_id", "Event ID", "Event Id", "id", "ID")
        event_name = first(row, "event_name", "Event Name", "event_title", "Event Title", "title", "Title")
        group_name = first(row, "group_name", "Group Name", "group", "Group")
        event_date_text = first(row, "event_date", "Event Date", "event_time", "Event Time", "date", "Date")
        rsvp_response = first(row, "rsvp_response", "RSVP Response", "response", "Response", "rsvp", "RSVP")
        venue_name = first(row, "venue_name", "Venue Name", "venue", "Venue")
        event_url = first(row, "event_url", "Event URL", "url", "URL", "link", "Link")
        location = first(row, "location", "Location", "venue_address", "Venue Address", "address", "Address")
        description = first(row, "description", "Description", "event_description", "Event Description")
        rsvp_date_text = first(row, "rsvp_date", "RSVP Date", "responded_at", "Responded At", "updated_at", "Updated At")

        if not any([event_id, event_name, group_name, event_date_text, rsvp_response, venue_name, event_url, location, description, rsvp_date_text]):
            return None

        event_at = parse_datetime(event_date_text)
        rsvp_at = parse_datetime(rsvp_date_text)
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "event_id": event_id,
                "event_name": event_name,
                "group_name": group_name,
                "event_date": event_at.isoformat() if event_at else event_date_text,
                "rsvp_response": rsvp_response,
                "venue_name": venue_name,
                "location": location,
                "event_url": event_url,
                "description": description,
                "rsvp_date": rsvp_at.isoformat() if rsvp_at else rsvp_date_text,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        timestamp = event_at or rsvp_at or now
        modified = rsvp_at or event_at or now
        return KnowledgeUnit(
            source_project="meetup_rsvps_csv",
            source_id=self._source_id(event_id, event_url, event_name, group_name, event_date_text, rsvp_response, index),
            source_entity_type="rsvp",
            title=event_name or "Meetup RSVP",
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["meetup", "rsvp", rsvp_response, group_name] if tag)),
            created_at=timestamp,
            updated_at=modified,
        )

    def _source_id(self, event_id: str, event_url: str, event_name: str, group_name: str, event_date: str, rsvp_response: str, index: int) -> str:
        if event_id:
            return f"meetup_rsvps_csv:{event_id}"
        if event_url:
            return digest_source_id("meetup_rsvps_csv", event_url)
        return digest_source_id("meetup_rsvps_csv", event_name, group_name, event_date, rsvp_response, index)

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            metadata.get("event_name", ""),
            metadata.get("description", ""),
            f"Group: {metadata.get('group_name')}" if metadata.get("group_name") else "",
            f"Date: {metadata.get('event_date')}" if metadata.get("event_date") else "",
            f"RSVP: {metadata.get('rsvp_response')}" if metadata.get("rsvp_response") else "",
            f"Venue: {metadata.get('venue_name')}" if metadata.get("venue_name") else "",
            f"Location: {metadata.get('location')}" if metadata.get("location") else "",
            f"URL: {metadata.get('event_url')}" if metadata.get("event_url") else "",
        ]
        return "\n".join(part for part in parts if part)
