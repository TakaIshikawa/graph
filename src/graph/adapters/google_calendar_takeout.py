"""Adapter for Google Calendar Takeout JSON event exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class GoogleCalendarTakeoutAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_calendar_takeout"

    @property
    def entity_types(self) -> list[str]:
        return ["event", "attendee"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or ["event"])
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        attendee_units: dict[str, KnowledgeUnit] = {}
        edge_candidates: list[KnowledgeEdge] = []
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
                if "event" in allowed_types:
                    units.append(unit)
                if "attendee" in allowed_types:
                    for attendee in self._attendees(event.get("attendees")):
                        attendee_unit = self._unit_from_attendee(attendee, unit.updated_at)
                        if attendee_unit is None:
                            continue
                        attendee_units.setdefault(attendee_unit.source_id, attendee_unit)
                        if "event" in allowed_types:
                            edge_candidates.append(self._edge_from_event_attendee(unit, attendee_unit, attendee))

        units.extend(attendee_units.values())
        result.units.extend(sorted(units, key=lambda unit: (unit.created_at, unit.source_id)))
        result.edges.extend(sorted(edge_candidates, key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id)))
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

    def _unit_from_attendee(
        self,
        attendee: dict[str, Any],
        event_updated_at: datetime,
    ) -> KnowledgeUnit | None:
        email = self._normalize_email(attendee.get("email"))
        display_name = self._string(attendee.get("displayName"))
        fallback = self._normalize_name(display_name)
        if not email and not fallback:
            return None

        title = display_name or email
        metadata = {
            "email": email,
            "display_name": display_name,
            "source": "google_calendar_attendee",
        }
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_CALENDAR_TAKEOUT,
            source_id=self._attendee_source_id(email, fallback),
            source_entity_type="attendee",
            title=title,
            content=title,
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["google_calendar", "person", "attendee"],
            created_at=event_updated_at,
            updated_at=event_updated_at,
        )

    def _edge_from_event_attendee(
        self,
        event_unit: KnowledgeUnit,
        attendee_unit: KnowledgeUnit,
        attendee: dict[str, Any],
    ) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._edge_id(event_unit.source_id, attendee_unit.source_id, attendee),
            from_unit_id=event_unit.source_id,
            to_unit_id=attendee_unit.source_id,
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.GOOGLE_CALENDAR_TAKEOUT.value,
                "from_entity_type": "event",
                "to_entity_type": "attendee",
                "response_status": attendee.get("responseStatus"),
                "organizer": attendee.get("organizer"),
                "self": attendee.get("self"),
                "attendee_email": self._normalize_email(attendee.get("email")),
                "attendee_display_name": self._string(attendee.get("displayName")),
            },
            created_at=event_unit.updated_at,
        )

    def _attendee_source_id(self, email: str, fallback: str) -> str:
        stable = f"email:{email}" if email else f"display:{fallback}"
        digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
        return f"google_calendar_takeout:attendee:{digest}"

    def _edge_id(
        self,
        event_source_id: str,
        attendee_source_id: str,
        attendee: dict[str, Any],
    ) -> str:
        raw = "|".join(
            [
                SourceProject.GOOGLE_CALENDAR_TAKEOUT.value,
                EdgeRelation.REFERENCES.value,
                event_source_id,
                attendee_source_id,
                self._string(attendee.get("responseStatus")),
            ]
        )
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"google-calendar-attendee-references-{digest}"

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

    def _string(self, value: Any) -> str:
        if value is None or isinstance(value, dict | list):
            return ""
        return str(value).strip()

    def _normalize_email(self, value: Any) -> str:
        return self._string(value).lower()

    def _normalize_name(self, value: Any) -> str:
        return " ".join(self._string(value).lower().split())

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
