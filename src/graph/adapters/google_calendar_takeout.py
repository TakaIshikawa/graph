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
        return ["event", "attendee", "attachment", "conference"]

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
        attachment_units: dict[str, KnowledgeUnit] = {}
        conference_units: dict[str, KnowledgeUnit] = {}
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
                if "attachment" in allowed_types:
                    for attachment in self._attachments(event):
                        attachment_unit = self._unit_from_attachment(attachment, unit, path.name)
                        attachment_units.setdefault(attachment_unit.source_id, attachment_unit)
                        if "event" in allowed_types:
                            edge_candidates.append(self._edge_from_event_child(unit, attachment_unit, "event_attachment"))
                if "conference" in allowed_types:
                    for conference in self._conferences(event):
                        conference_unit = self._unit_from_conference(conference, unit, path.name)
                        conference_units.setdefault(conference_unit.source_id, conference_unit)
                        if "event" in allowed_types:
                            edge_candidates.append(self._edge_from_event_child(unit, conference_unit, "event_conference"))

        units.extend(attendee_units.values())
        units.extend(attachment_units.values())
        units.extend(conference_units.values())
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

    def _attachments(self, event: dict[str, Any]) -> list[dict[str, str]]:
        raw = event.get("attachments")
        if not isinstance(raw, list):
            return []
        attachments: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            url = self._first(item, "fileUrl", "url", "alternateLink")
            title = self._first(item, "title", "fileName", "name") or url
            if not title and not url:
                continue
            attachments.append(
                {
                    "title": title,
                    "url": url,
                    "mime_type": self._first(item, "mimeType", "mime_type"),
                    "file_id": self._first(item, "fileId", "id"),
                    "icon_link": self._first(item, "iconLink"),
                }
            )
        return attachments

    def _unit_from_attachment(
        self,
        attachment: dict[str, str],
        event_unit: KnowledgeUnit,
        source_file: str,
    ) -> KnowledgeUnit:
        source_id = self._child_source_id("attachment", event_unit.source_id, attachment.get("file_id") or attachment.get("url") or attachment.get("title"))
        title = attachment.get("title") or "Calendar attachment"
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_CALENDAR_TAKEOUT,
            source_id=source_id,
            source_entity_type="attachment",
            title=title,
            content=self._attachment_content(attachment),
            content_type=ContentType.ARTIFACT,
            metadata={
                "title": attachment.get("title"),
                "url": attachment.get("url"),
                "mime_type": attachment.get("mime_type"),
                "file_id": attachment.get("file_id"),
                "icon_link": attachment.get("icon_link"),
                "event_source_id": event_unit.source_id,
                "source_file": source_file,
            },
            tags=["google_calendar", "attachment"],
            created_at=event_unit.created_at,
            updated_at=event_unit.updated_at,
        )

    def _conferences(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        conferences: list[dict[str, Any]] = []
        data = event.get("conferenceData")
        if isinstance(data, dict):
            entry_points = [
                item for item in data.get("entryPoints", []) if isinstance(item, dict)
            ] if isinstance(data.get("entryPoints"), list) else []
            conferences.append(
                {
                    "provider": self._conference_provider(data),
                    "meeting_code": self._conference_code(data),
                    "url": self._first_entry_url(entry_points),
                    "entry_points": entry_points,
                    "conference_id": self._first(data, "conferenceId", "signature"),
                }
            )
        hangout_link = self._first(event, "hangoutLink")
        if hangout_link and all(item.get("url") != hangout_link for item in conferences):
            conferences.append({"provider": "hangouts", "meeting_code": "", "url": hangout_link, "entry_points": [], "conference_id": hangout_link})
        return [item for item in conferences if item.get("url") or item.get("meeting_code") or item.get("conference_id")]

    def _unit_from_conference(
        self,
        conference: dict[str, Any],
        event_unit: KnowledgeUnit,
        source_file: str,
    ) -> KnowledgeUnit:
        source_id = self._child_source_id("conference", event_unit.source_id, conference.get("conference_id") or conference.get("url") or conference.get("meeting_code"))
        title = conference.get("provider") or "Calendar conference"
        if conference.get("meeting_code"):
            title = f"{title} {conference['meeting_code']}"
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_CALENDAR_TAKEOUT,
            source_id=source_id,
            source_entity_type="conference",
            title=title,
            content=self._conference_content(conference),
            content_type=ContentType.METADATA,
            metadata={
                "provider": conference.get("provider"),
                "meeting_code": conference.get("meeting_code"),
                "url": conference.get("url"),
                "entry_points": conference.get("entry_points"),
                "event_source_id": event_unit.source_id,
                "source_file": source_file,
            },
            tags=["google_calendar", "conference"],
            created_at=event_unit.created_at,
            updated_at=event_unit.updated_at,
        )

    def _edge_from_event_child(
        self,
        event_unit: KnowledgeUnit,
        child_unit: KnowledgeUnit,
        relation_type: str,
    ) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._child_edge_id(event_unit.source_id, child_unit.source_id, relation_type),
            from_unit_id=event_unit.source_id,
            to_unit_id=child_unit.source_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.GOOGLE_CALENDAR_TAKEOUT.value,
                "relation_type": relation_type,
                "from_entity_type": "event",
                "to_entity_type": child_unit.source_entity_type,
            },
            created_at=event_unit.updated_at,
        )

    def _child_source_id(self, kind: str, event_source_id: str, stable: Any) -> str:
        digest = hashlib.sha256(f"{event_source_id}|{kind}|{stable}".encode("utf-8")).hexdigest()[:24]
        return f"google_calendar_takeout:{kind}:{digest}"

    def _child_edge_id(self, event_source_id: str, child_source_id: str, relation_type: str) -> str:
        digest = hashlib.sha1(f"{event_source_id}|{child_source_id}|{relation_type}".encode("utf-8")).hexdigest()[:16]
        return f"google-calendar-{relation_type}-{digest}"

    def _conference_provider(self, data: dict[str, Any]) -> str:
        solution = data.get("conferenceSolution")
        if isinstance(solution, dict):
            return self._first(solution, "name", "key")
        return self._first(data, "conferenceSolution", "provider", "type")

    def _conference_code(self, data: dict[str, Any]) -> str:
        return self._first(data, "conferenceId", "meetingCode", "meeting_code")

    def _first_entry_url(self, entry_points: list[dict[str, Any]]) -> str:
        for item in entry_points:
            url = self._first(item, "uri", "url")
            if url:
                return url
        return ""

    def _attachment_content(self, attachment: dict[str, str]) -> str:
        parts = [attachment.get("title") or "Calendar attachment"]
        if attachment.get("url"):
            parts.append(f"URL: {attachment['url']}")
        if attachment.get("mime_type"):
            parts.append(f"MIME type: {attachment['mime_type']}")
        return "\n".join(parts)

    def _conference_content(self, conference: dict[str, Any]) -> str:
        parts = [str(conference.get("provider") or "Calendar conference")]
        if conference.get("meeting_code"):
            parts.append(f"Meeting code: {conference['meeting_code']}")
        if conference.get("url"):
            parts.append(f"URL: {conference['url']}")
        return "\n".join(parts)

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
