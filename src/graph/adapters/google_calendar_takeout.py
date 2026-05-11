"""Adapter for Google Calendar Takeout JSON event exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GoogleCalendarTakeoutAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_calendar_takeout"

    @property
    def entity_types(self) -> list[str]:
        return ["event"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "event" not in entity_types:
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                events = self._read_events(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for event, calendar_name in events:
                unit = self._unit_from_event(event, calendar_name, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.created_at, unit.source_id)))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.json") if path.is_file())

    def _read_events(self, path: Path) -> list[tuple[dict[str, Any], str]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [(event, path.stem) for event in parsed if isinstance(event, dict)]
        if not isinstance(parsed, dict):
            return []

        calendar_name = self._first(parsed, "summary", "calendarName", "name", "title") or path.stem
        raw_events = parsed.get("items")
        if not isinstance(raw_events, list):
            raw_events = parsed.get("events")
        if isinstance(raw_events, list):
            return [(event, calendar_name) for event in raw_events if isinstance(event, dict)]
        return []

    def _unit_from_event(
        self,
        event: dict[str, Any],
        calendar_name: str,
        source_file: str,
    ) -> KnowledgeUnit | None:
        status = self._first(event, "status").lower()
        if status == "cancelled":
            return None

        start = self._date_value(event.get("start"))
        end = self._date_value(event.get("end"))
        start_at = self._date_datetime(start)
        if start_at is None:
            return None

        updated_at = self._parse_datetime(self._first(event, "updated", "modified")) or start_at
        created_at = self._parse_datetime(self._first(event, "created")) or start_at
        title = self._first(event, "summary", "title", "name") or "Untitled calendar event"
        description = self._first(event, "description", "notes")
        location = self._first(event, "location")
        html_link = self._first(event, "htmlLink", "url")
        event_id = self._first(event, "id", "iCalUID", "uid")

        metadata = {
            "event_id": event_id,
            "calendar_name": calendar_name,
            "source_url": html_link,
            "description": description,
            "location": location,
            "attendees": self._attendees(event.get("attendees")),
            "start": start,
            "end": end,
            "status": status or None,
            "created": self._first(event, "created"),
            "updated": self._first(event, "updated", "modified"),
            "source_file": source_file,
        }

        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_CALENDAR_TAKEOUT,
            source_id=self._source_id(event, event_id, title, start, end, calendar_name),
            source_entity_type="event",
            title=title,
            content=self._content(title, description, location, start, end, html_link),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["google_calendar", "calendar_event"],
            created_at=created_at,
            updated_at=updated_at,
        )

    def _source_id(
        self,
        event: dict[str, Any],
        event_id: str,
        title: str,
        start: dict[str, str],
        end: dict[str, str],
        calendar_name: str,
    ) -> str:
        stable = event_id or json.dumps(
            {
                "calendar_name": calendar_name,
                "title": title,
                "start": start,
                "end": end,
                "location": self._first(event, "location"),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
        return f"google_calendar_takeout:event:{digest}"

    def _date_value(self, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        if value.get("dateTime"):
            result = {"dateTime": str(value["dateTime"])}
            if value.get("timeZone"):
                result["timeZone"] = str(value["timeZone"])
            return result
        if value.get("date"):
            return {"date": str(value["date"])}
        return {}

    def _date_datetime(self, value: dict[str, str]) -> datetime | None:
        raw = value.get("dateTime") or value.get("date")
        if not raw:
            return None
        if "dateTime" in value:
            return self._parse_datetime(raw)
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone.utc)

    def _attendees(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        attendees: list[dict[str, Any]] = []
        for attendee in value:
            if not isinstance(attendee, dict):
                continue
            attendees.append(
                {
                    "email": attendee.get("email"),
                    "displayName": attendee.get("displayName"),
                    "responseStatus": attendee.get("responseStatus"),
                    "organizer": attendee.get("organizer"),
                    "self": attendee.get("self"),
                }
            )
        return attendees

    def _content(
        self,
        title: str,
        description: str,
        location: str,
        start: dict[str, str],
        end: dict[str, str],
        html_link: str,
    ) -> str:
        parts = [title]
        if description:
            parts.append(description)
        if location:
            parts.append(f"Location: {location}")
        if start:
            parts.append(f"Start: {start.get('dateTime') or start.get('date')}")
        if end:
            parts.append(f"End: {end.get('dateTime') or end.get('date')}")
        if html_link:
            parts.append(f"URL: {html_link}")
        return "\n".join(parts)

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return self._ensure_utc(parsed)

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
