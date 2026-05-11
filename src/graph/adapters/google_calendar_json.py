"""Adapter for Google Calendar JSON event exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GoogleCalendarJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_calendar_json"

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
        sync_at = self._sync_datetime(since) if since else None

        for path in self._iter_paths():
            try:
                events = self._read_events(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for event in events:
                unit = self._unit_from_event(event)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _read_events(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [event for event in parsed if isinstance(event, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
            return [event for event in parsed["items"] if isinstance(event, dict)]
        return []

    def _unit_from_event(self, event: dict[str, Any]) -> KnowledgeUnit | None:
        event_id = self._first(event, "id", "iCalUID")
        title = self._first(event, "summary") or "Untitled calendar event"
        description = self._first(event, "description")
        html_link = self._first(event, "htmlLink")
        start = self._date_value(event.get("start"))
        end = self._date_value(event.get("end"))
        updated = self._parse_datetime(self._first(event, "updated"))
        created = self._parse_datetime(self._first(event, "created"))
        comparable = updated or self._parse_datetime(start.get("dateTime", "")) or created
        now = datetime.now(timezone.utc)
        metadata = {
            "event_id": event_id,
            "source_url": html_link,
            "location": self._first(event, "location"),
            "attendees": self._attendees(event.get("attendees")),
            "start": start,
            "end": end,
            "status": self._first(event, "status"),
            "created": self._first(event, "created"),
            "updated": self._first(event, "updated"),
        }
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_CALENDAR_JSON,
            source_id=f"google_calendar_json:{event_id or html_link or title}",
            source_entity_type="event",
            title=title,
            content=self._content(title, description, html_link, start, end),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["google_calendar"],
            created_at=created or comparable or now,
            updated_at=updated or comparable or created or now,
        )

    def _date_value(self, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        if value.get("dateTime"):
            return {"dateTime": str(value["dateTime"]), "timeZone": str(value.get("timeZone", ""))}
        if value.get("date"):
            return {"date": str(value["date"])}
        return {}

    def _attendees(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        attendees = []
        for attendee in value:
            if isinstance(attendee, dict):
                attendees.append(
                    {
                        "email": attendee.get("email"),
                        "displayName": attendee.get("displayName"),
                        "responseStatus": attendee.get("responseStatus"),
                    }
                )
        return attendees

    def _content(
        self,
        title: str,
        description: str,
        html_link: str,
        start: dict[str, str],
        end: dict[str, str],
    ) -> str:
        parts = [title]
        if description:
            parts.append(description)
        if start:
            parts.append(f"Start: {start.get('dateTime') or start.get('date')}")
        if end:
            parts.append(f"End: {end.get('dateTime') or end.get('date')}")
        if html_link:
            parts.append(f"URL: {html_link}")
        return "\n".join(parts)

    def _iter_paths(self) -> list[Path]:
        path = Path(self.path).expanduser() if self.path else None
        if path is None:
            return []
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(child for child in path.rglob("*.json") if child.is_file())
        return []

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
