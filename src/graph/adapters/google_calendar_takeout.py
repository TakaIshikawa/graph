"""Adapter for Google Calendar Takeout JSON event exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
                        if attachment_unit is None:
                            continue
                        attachment_units.setdefault(attachment_unit.source_id, attachment_unit)
                        if "event" in allowed_types:
                            edge_candidates.append(
                                self._edge_from_event_resource(unit, attachment_unit, "attachment", attachment)
                            )
                if "conference" in allowed_types:
                    for conference in self._conferences(event):
                        conference_unit = self._unit_from_conference(conference, unit, path.name)
                        if conference_unit is None:
                            continue
                        conference_units.setdefault(conference_unit.source_id, conference_unit)
                        if "event" in allowed_types:
                            edge_candidates.append(
                                self._edge_from_event_resource(unit, conference_unit, "conference", conference)
                            )

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

    def _attachments(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        raw = value.get("attachments")
        if not isinstance(raw, list):
            return []
        attachments: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            url = self._first(item, "fileUrl", "url", "alternateLink")
            file_id = self._first(item, "fileId", "id")
            title = self._first(item, "title", "fileName", "name") or url or file_id
            stable = file_id or url or title
            if not stable or stable in seen:
                continue
            seen.add(stable)
            attachments.append(
                {
                    "file_id": file_id,
                    "title": title,
                    "mime_type": self._first(item, "mimeType", "mime_type"),
                    "url": url,
                    "icon_link": self._first(item, "iconLink", "icon"),
                }
            )
        return attachments

    def _conferences(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        conferences: list[dict[str, Any]] = []
        data = event.get("conferenceData")
        if isinstance(data, dict):
            solution = data.get("conferenceSolution")
            solution_name = ""
            solution_key = ""
            if isinstance(solution, dict):
                solution_name = self._first(solution, "name")
                key = solution.get("key")
                if isinstance(key, dict):
                    solution_key = self._first(key, "type")
            entry_points = self._entry_points(data.get("entryPoints"))
            conference_id = self._first(data, "conferenceId", "id", "signature")
            url = self._first(data, "hangoutLink")
            if not url:
                url = next((entry["uri"] for entry in entry_points if entry.get("uri")), "")
            if conference_id or url or entry_points:
                conferences.append(
                    {
                        "provider": solution_name if solution_key else self._normalize_provider(solution_name or solution_key),
                        "provider_type": solution_key,
                        "meeting_code": conference_id,
                        "url": url,
                        "entry_points": entry_points,
                    }
                )

        hangout_link = self._first(event, "hangoutLink")
        if hangout_link and all(item.get("url") != hangout_link for item in conferences):
            conferences.append(
                {
                    "provider": self._conference_provider(hangout_link),
                    "provider_type": "",
                    "meeting_code": self._meeting_code(hangout_link),
                    "url": hangout_link,
                    "entry_points": [{"entryPointType": "video", "uri": hangout_link}],
                }
            )

        location = self._first(event, "location")
        for location_url in self._urls(location):
            if any(item.get("url") == location_url for item in conferences):
                continue
            conferences.append(
                {
                    "provider": self._conference_provider(location_url),
                    "provider_type": "",
                    "meeting_code": self._meeting_code(location_url),
                    "url": location_url,
                    "entry_points": [{"entryPointType": "location_url", "uri": location_url}],
                }
            )
        return conferences

    def _entry_points(self, value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        entries: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            entry = {
                "entryPointType": self._first(item, "entryPointType"),
                "uri": self._first(item, "uri"),
                "label": self._first(item, "label"),
                "meetingCode": self._first(item, "meetingCode"),
                "passcode": self._first(item, "passcode", "password"),
            }
            if any(entry.values()):
                entries.append({key: value for key, value in entry.items() if value})
        return entries

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

    def _unit_from_attachment(
        self,
        attachment: dict[str, Any],
        event_unit: KnowledgeUnit,
        source_file: str,
    ) -> KnowledgeUnit | None:
        stable = attachment.get("file_id") or attachment.get("url") or attachment.get("title")
        if not stable:
            return None
        title = str(attachment.get("title") or attachment.get("url") or "Calendar attachment")
        metadata = {
            **attachment,
            "parent_event_source_id": event_unit.source_id,
            "parent_event_title": event_unit.title,
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_CALENDAR_TAKEOUT,
            source_id=self._resource_source_id("attachment", str(stable)),
            source_entity_type="attachment",
            title=title,
            content=self._resource_content(title, attachment.get("url"), attachment.get("mime_type")),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=["google_calendar", "calendar_attachment"],
            created_at=event_unit.created_at,
            updated_at=event_unit.updated_at,
        )

    def _unit_from_conference(
        self,
        conference: dict[str, Any],
        event_unit: KnowledgeUnit,
        source_file: str,
    ) -> KnowledgeUnit | None:
        stable = conference.get("meeting_code") or conference.get("url") or json.dumps(conference, sort_keys=True)
        if not stable:
            return None
        provider = str(conference.get("provider") or "Conference")
        meeting_code = str(conference.get("meeting_code") or "")
        title = f"{provider} {meeting_code}".strip()
        metadata = {
            **conference,
            "parent_event_source_id": event_unit.source_id,
            "parent_event_title": event_unit.title,
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_CALENDAR_TAKEOUT,
            source_id=self._resource_source_id("conference", str(stable)),
            source_entity_type="conference",
            title=title,
            content=self._resource_content(title, conference.get("url"), meeting_code),
            content_type=ContentType.METADATA,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=["google_calendar", "calendar_conference"],
            created_at=event_unit.created_at,
            updated_at=event_unit.updated_at,
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

    def _edge_from_event_resource(
        self,
        event_unit: KnowledgeUnit,
        target_unit: KnowledgeUnit,
        target_type: str,
        raw: dict[str, Any],
    ) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._resource_edge_id(event_unit.source_id, target_unit.source_id, target_type),
            from_unit_id=event_unit.source_id,
            to_unit_id=target_unit.source_id,
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.GOOGLE_CALENDAR_TAKEOUT.value,
                "from_entity_type": "event",
                "to_entity_type": target_type,
                "url": raw.get("url"),
                "meeting_code": raw.get("meeting_code"),
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

    def _resource_source_id(self, entity_type: str, stable: str) -> str:
        digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
        return f"google_calendar_takeout:{entity_type}:{digest}"

    def _resource_edge_id(self, event_source_id: str, target_source_id: str, target_type: str) -> str:
        raw = "|".join([event_source_id, target_source_id, target_type, EdgeRelation.REFERENCES.value])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"google-calendar-{target_type}-references-{digest}"

    def _resource_content(self, title: str, url: Any, detail: Any) -> str:
        parts = [title]
        if detail:
            parts.append(str(detail))
        if url:
            parts.append(f"URL: {url}")
        return "\n".join(parts)

    def _is_url(self, value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _conference_provider(self, url: str) -> str:
        host = urlparse(url).netloc.lower()
        if "meet.google" in host or "hangouts.google" in host:
            return "Google Meet"
        if "zoom.us" in host:
            return "Zoom"
        if "teams.microsoft" in host:
            return "Microsoft Teams"
        return host or "Conference"

    def _normalize_provider(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")

    def _meeting_code(self, url: str) -> str:
        path = urlparse(url).path.strip("/")
        return path.rsplit("/", 1)[-1] if path else ""

    def _urls(self, value: str) -> list[str]:
        return re.findall(r"https?://[^\s,)>]+", value or "")

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
