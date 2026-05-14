"""Adapter for Google Calendar JSON event exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class GoogleCalendarJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_calendar_json"

    @property
    def entity_types(self) -> list[str]:
        return ["event", "person"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types) if entity_types is not None else {"event"}
        if not requested.intersection(self.entity_types):
            return result
        sync_at = self._sync_datetime(since) if since else None
        event_units: list[KnowledgeUnit] = []

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
                event_units.append(unit)
                if "event" in requested:
                    result.units.append(unit)
                    result.edges.extend(self._participant_edges(unit))
        if "person" in requested:
            result.units.extend(self._person_units(event_units))

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        result.edges = sorted({edge.id: edge for edge in result.edges}.values(), key=lambda edge: edge.id)
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
        organizer = self._person(event.get("organizer"))
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
        if any(value is not None for value in organizer.values()):
            metadata["organizer"] = organizer
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
                attendees.append(self._person(attendee, include_response=True))
        return attendees

    def _person(self, value: Any, *, include_response: bool = False) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        person = {
            "email": value.get("email"),
            "displayName": value.get("displayName"),
        }
        if include_response:
            person["responseStatus"] = value.get("responseStatus")
        return person

    def _participant_edges(self, unit: KnowledgeUnit) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        organizer = unit.metadata.get("organizer")
        if isinstance(organizer, dict):
            edge = self._participant_edge(unit.source_id, organizer, "organizer")
            if edge:
                edges.append(edge)
        for attendee in unit.metadata.get("attendees", []):
            if not isinstance(attendee, dict):
                continue
            edge = self._participant_edge(unit.source_id, attendee, "attendee")
            if edge:
                edges.append(edge)
        return edges

    def _participant_edge(self, source_id: str, person: dict[str, Any], kind: str) -> KnowledgeEdge | None:
        email = self._normalize_email(person.get("email"))
        if not email:
            return None
        target_id = self._person_source_id(email)
        digest = hashlib.sha256(f"{source_id}|participant|{target_id}".encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(
            id=f"google_calendar_json:participant:{digest}",
            from_unit_id=source_id,
            to_unit_id=target_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={
                "kind": kind,
                "email": email,
                "displayName": person.get("displayName"),
                "responseStatus": person.get("responseStatus"),
                "source_project": SourceProject.GOOGLE_CALENDAR_JSON.value,
            },
        )

    def _person_units(self, events: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, dict[str, Any]] = {}
        for event in events:
            participants: list[tuple[str, dict[str, Any]]] = []
            organizer = event.metadata.get("organizer")
            if isinstance(organizer, dict):
                participants.append(("organizer", organizer))
            for attendee in event.metadata.get("attendees", []):
                if isinstance(attendee, dict):
                    participants.append(("attendee", attendee))
            for role, person in participants:
                email = self._normalize_email(person.get("email"))
                if not email:
                    continue
                info = grouped.setdefault(
                    email,
                    {
                        "events": [],
                        "display_names": set(),
                        "response_statuses": set(),
                        "organizer_count": 0,
                        "attendee_count": 0,
                    },
                )
                info["events"].append(event)
                display_name = person.get("displayName")
                if display_name:
                    info["display_names"].add(str(display_name))
                response_status = person.get("responseStatus")
                if response_status:
                    info["response_statuses"].add(str(response_status))
                if role == "organizer":
                    info["organizer_count"] += 1
                else:
                    info["attendee_count"] += 1

        units: list[KnowledgeUnit] = []
        now = datetime.now(timezone.utc)
        for email, info in grouped.items():
            event_source_ids = sorted({event.source_id for event in info["events"]})
            created_at = min((event.created_at for event in info["events"]), default=now)
            updated_at = max((event.updated_at for event in info["events"]), default=created_at)
            display_names = sorted(info["display_names"])
            metadata = {
                "email": email,
                "display_names": display_names,
                "response_statuses": sorted(info["response_statuses"]),
                "event_source_ids": event_source_ids,
                "event_count": len(event_source_ids),
                "organizer_count": info["organizer_count"],
                "attendee_count": info["attendee_count"],
            }
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GOOGLE_CALENDAR_JSON,
                    source_id=self._person_source_id(email),
                    source_entity_type="person",
                    title=display_names[0] if display_names else email,
                    content=f"Google Calendar person: {display_names[0] if display_names else email}\nEvents: {len(event_source_ids)}",
                    content_type=ContentType.METADATA,
                    metadata={key: value for key, value in metadata.items() if value not in ("", None, [], set())},
                    tags=["google_calendar", "person"],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _person_source_id(self, email: str) -> str:
        return f"google_calendar:person:{email}"

    def _normalize_email(self, value: Any) -> str:
        return str(value).strip().casefold() if value not in ("", None) else ""

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
