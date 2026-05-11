from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.google_calendar_takeout import GoogleCalendarTakeoutAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def _write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_google_calendar_takeout_imports_file_items_with_event_metadata(tmp_path):
    export = _write_json(
        tmp_path / "Work.json",
        {
            "summary": "Work Calendar",
            "items": [
                {
                    "id": "event-1",
                    "summary": "Design review",
                    "description": "Review the import flows",
                    "htmlLink": "https://calendar.google.com/event?eid=1",
                    "location": "Room 1",
                    "start": {"dateTime": "2026-05-01T10:00:00+09:00", "timeZone": "Asia/Tokyo"},
                    "end": {"dateTime": "2026-05-01T11:00:00+09:00", "timeZone": "Asia/Tokyo"},
                    "attendees": [
                        {
                            "email": "ada@example.com",
                            "displayName": "Ada",
                            "responseStatus": "accepted",
                            "organizer": True,
                        }
                    ],
                    "status": "confirmed",
                    "created": "2026-04-20T00:00:00Z",
                    "updated": "2026-04-21T00:00:00Z",
                }
            ],
        },
    )

    unit = GoogleCalendarTakeoutAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == SourceProject.GOOGLE_CALENDAR_TAKEOUT
    assert unit.source_entity_type == "event"
    assert unit.title == "Design review"
    assert "Review the import flows" in unit.content
    assert unit.created_at == datetime(2026, 4, 20, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 4, 21, tzinfo=timezone.utc)
    assert unit.metadata["calendar_name"] == "Work Calendar"
    assert unit.metadata["location"] == "Room 1"
    assert unit.metadata["start"] == {"dateTime": "2026-05-01T10:00:00+09:00", "timeZone": "Asia/Tokyo"}
    assert unit.metadata["end"] == {"dateTime": "2026-05-01T11:00:00+09:00", "timeZone": "Asia/Tokyo"}
    assert unit.metadata["attendees"] == [
        {
            "email": "ada@example.com",
            "displayName": "Ada",
            "responseStatus": "accepted",
            "organizer": True,
            "self": None,
        }
    ]
    assert unit.metadata["status"] == "confirmed"


def test_google_calendar_takeout_imports_directory_and_events_key(tmp_path):
    _write_json(
        tmp_path / "Personal.json",
        {
            "calendarName": "Personal",
            "events": [
                {
                    "id": "all-day",
                    "summary": "Away",
                    "start": {"date": "2026-05-03"},
                    "end": {"date": "2026-05-04"},
                }
            ],
        },
    )
    _write_json(
        tmp_path / "nested" / "Team.json",
        {
            "summary": "Team",
            "items": [
                {
                    "id": "date-time",
                    "summary": "Planning",
                    "start": {"dateTime": "2026-05-02T09:00:00Z"},
                    "end": {"dateTime": "2026-05-02T10:00:00Z"},
                }
            ],
        },
    )

    result = GoogleCalendarTakeoutAdapter(path=str(tmp_path)).ingest()

    assert [unit.title for unit in result.units] == ["Planning", "Away"]
    away = next(unit for unit in result.units if unit.title == "Away")
    assert away.created_at == datetime(2026, 5, 3, tzinfo=timezone.utc)
    assert away.updated_at == datetime(2026, 5, 3, tzinfo=timezone.utc)
    assert away.metadata["calendar_name"] == "Personal"
    assert away.metadata["start"] == {"date": "2026-05-03"}


def test_google_calendar_takeout_skips_cancelled_and_malformed_events(tmp_path):
    export = _write_json(
        tmp_path / "calendar.json",
        {
            "items": [
                {
                    "id": "cancelled",
                    "summary": "Cancelled",
                    "status": "cancelled",
                    "start": {"dateTime": "2026-05-01T09:00:00Z"},
                },
                {"id": "missing-start", "summary": "Malformed"},
                {
                    "id": "ok",
                    "summary": "Confirmed",
                    "start": {"dateTime": "2026-05-01T10:00:00Z"},
                },
            ]
        },
    )

    result = GoogleCalendarTakeoutAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == ["Confirmed"]


def test_google_calendar_takeout_filters_entity_types_and_since(tmp_path):
    export = _write_json(
        tmp_path / "calendar.json",
        {
            "items": [
                {
                    "id": "old",
                    "summary": "Old",
                    "start": {"dateTime": "2026-05-01T09:00:00Z"},
                    "updated": "2026-05-01T09:30:00Z",
                },
                {
                    "id": "new",
                    "summary": "New",
                    "start": {"dateTime": "2026-05-02T09:00:00Z"},
                    "updated": "2026-05-02T09:30:00Z",
                },
            ]
        },
    )
    adapter = GoogleCalendarTakeoutAdapter(path=str(export))

    assert adapter.ingest(entity_types=["note"]).units == []

    result = adapter.ingest(
        entity_types=["event"],
        since=SyncState(
            source_project="google_calendar_takeout",
            source_entity_type="event",
            last_sync_at=datetime(2026, 5, 1, 12, tzinfo=timezone.utc),
        ),
    )

    assert [unit.title for unit in result.units] == ["New"]


def test_google_calendar_takeout_reports_event_and_attendee_entity_types():
    assert GoogleCalendarTakeoutAdapter().entity_types == ["event", "attendee"]


def test_google_calendar_takeout_imports_attendees_and_event_edges_when_requested(tmp_path):
    export = _write_json(
        tmp_path / "Work.json",
        {
            "summary": "Work Calendar",
            "items": [
                {
                    "id": "event-1",
                    "summary": "Design review",
                    "start": {"dateTime": "2026-05-01T10:00:00Z"},
                    "updated": "2026-05-01T11:00:00Z",
                    "attendees": [
                        {
                            "email": "Ada@Example.com ",
                            "displayName": "Ada Lovelace",
                            "responseStatus": "accepted",
                            "organizer": True,
                        },
                        {
                            "displayName": "Grace Hopper",
                            "responseStatus": "needsAction",
                        },
                    ],
                }
            ],
        },
    )

    result = GoogleCalendarTakeoutAdapter(path=str(export)).ingest(entity_types=["event", "attendee"])

    assert [unit.source_entity_type for unit in result.units] == ["event", "attendee", "attendee"]
    event_unit = next(unit for unit in result.units if unit.source_entity_type == "event")
    attendees = {unit.title: unit for unit in result.units if unit.source_entity_type == "attendee"}
    ada = attendees["Ada Lovelace"]
    grace = attendees["Grace Hopper"]
    assert ada.source_id.startswith("google_calendar_takeout:attendee:")
    assert ada.metadata == {
        "email": "ada@example.com",
        "display_name": "Ada Lovelace",
        "source": "google_calendar_attendee",
    }
    assert grace.metadata["email"] == ""

    assert len(result.edges) == 2
    assert {(edge.from_unit_id, edge.to_unit_id) for edge in result.edges} == {
        (event_unit.source_id, ada.source_id),
        (event_unit.source_id, grace.source_id),
    }
    assert {edge.relation for edge in result.edges} == {EdgeRelation.REFERENCES}
    assert {edge.source for edge in result.edges} == {EdgeSource.SOURCE}
    edge_by_attendee = {edge.to_unit_id: edge for edge in result.edges}
    assert edge_by_attendee[ada.source_id].metadata["response_status"] == "accepted"
    assert edge_by_attendee[ada.source_id].metadata["attendee_email"] == "ada@example.com"
    assert edge_by_attendee[grace.source_id].metadata["response_status"] == "needsAction"


def test_google_calendar_takeout_attendee_source_ids_are_stable(tmp_path):
    export = _write_json(
        tmp_path / "calendar.json",
        {
            "items": [
                {
                    "id": "one",
                    "summary": "One",
                    "start": {"dateTime": "2026-05-01T10:00:00Z"},
                    "attendees": [
                        {"email": "ADA@EXAMPLE.COM", "displayName": "Ada"},
                        {"displayName": "  Grace   Hopper  "},
                    ],
                },
                {
                    "id": "two",
                    "summary": "Two",
                    "start": {"dateTime": "2026-05-02T10:00:00Z"},
                    "attendees": [
                        {"email": "ada@example.com", "displayName": "A. Lovelace"},
                        {"displayName": "grace hopper"},
                    ],
                },
            ]
        },
    )

    result = GoogleCalendarTakeoutAdapter(path=str(export)).ingest(entity_types=["attendee"])

    attendees = sorted(result.units, key=lambda unit: unit.metadata["display_name"])
    assert len(attendees) == 2
    assert {unit.metadata["email"] for unit in attendees} == {"", "ada@example.com"}
    first_ids = {unit.source_id for unit in attendees}
    second_ids = {
        unit.source_id
        for unit in GoogleCalendarTakeoutAdapter(path=str(export)).ingest(entity_types=["attendee"]).units
    }
    assert first_ids == second_ids
    assert result.edges == []


def test_google_calendar_takeout_entity_type_filters_preserve_valid_edges(tmp_path):
    export = _write_json(
        tmp_path / "calendar.json",
        {
            "items": [
                {
                    "id": "event-1",
                    "summary": "Design review",
                    "start": {"dateTime": "2026-05-01T10:00:00Z"},
                    "attendees": [{"email": "ada@example.com", "displayName": "Ada"}],
                }
            ]
        },
    )
    adapter = GoogleCalendarTakeoutAdapter(path=str(export))

    default_result = adapter.ingest()
    event_result = adapter.ingest(entity_types=["event"])
    attendee_result = adapter.ingest(entity_types=["attendee"])
    combined_result = adapter.ingest(entity_types=["event", "attendee"])

    assert [unit.source_entity_type for unit in default_result.units] == ["event"]
    assert default_result.edges == []
    assert [unit.source_entity_type for unit in event_result.units] == ["event"]
    assert event_result.edges == []
    assert [unit.source_entity_type for unit in attendee_result.units] == ["attendee"]
    assert attendee_result.edges == []
    assert sorted(unit.source_entity_type for unit in combined_result.units) == ["attendee", "event"]
    assert len(combined_result.edges) == 1


def test_google_calendar_takeout_uses_stable_source_ids_without_event_ids(tmp_path):
    event = {
        "summary": "No ID",
        "location": "Room 2",
        "start": {"dateTime": "2026-05-05T09:00:00Z"},
        "end": {"dateTime": "2026-05-05T10:00:00Z"},
    }
    export = _write_json(tmp_path / "calendar.json", {"summary": "Calendar", "items": [event]})

    first = GoogleCalendarTakeoutAdapter(path=str(export)).ingest().units[0]
    second = GoogleCalendarTakeoutAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("google_calendar_takeout:event:")


def test_google_calendar_takeout_adapter_is_registered():
    assert isinstance(
        get_adapter("google_calendar_takeout", path="/tmp/calendar.json"),
        GoogleCalendarTakeoutAdapter,
    )
