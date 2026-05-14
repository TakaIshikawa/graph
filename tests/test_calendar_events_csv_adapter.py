from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.calendar_events_csv import CalendarEventsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_calendar_events_csv_ingests_common_headers(tmp_path):
    export = tmp_path / "events.csv"
    _write_csv(
        export,
        [
            {
                "Subject": "Planning",
                "Description": "Roadmap discussion",
                "Where": "Room 1",
                "Start Time": "2026-05-01 10:00",
                "End Time": "2026-05-01 11:00",
                "Attendees": "a@example.com; b@example.com",
                "Organizer": "owner@example.com",
                "URL": "https://meet.example/event",
            }
        ],
    )

    result = CalendarEventsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.CALENDAR_EVENTS_CSV
    assert unit.source_entity_type == "calendar_event"
    assert unit.title == "Planning"
    assert unit.metadata["location"] == "Room 1"
    assert unit.metadata["start_at"] == "2026-05-01T10:00:00+00:00"
    assert unit.metadata["end_at"] == "2026-05-01T11:00:00+00:00"
    assert unit.metadata["attendees"] == ["a@example.com", "b@example.com"]
    assert unit.metadata["organizer"] == "owner@example.com"
    assert unit.updated_at == datetime(2026, 5, 1, 10, tzinfo=timezone.utc)
    assert "URL: https://meet.example/event" in unit.content


def test_calendar_events_csv_directory_since_entity_filter_all_day_and_invalid_dates(tmp_path):
    _write_csv(
        tmp_path / "events.csv",
        [
            {"Title": "Old", "Start": "2026-04-30"},
            {"Summary": "All Day", "Start Date": "2026-05-03", "End Date": "2026-05-04", "All Day": "yes"},
            {"Event": "", "Description": "Missing title", "Start": "not a date", "Location": "Desk"},
        ],
    )
    since = SyncState(
        source_project="calendar_events_csv",
        source_entity_type="calendar_event",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = CalendarEventsCsvAdapter(path=str(tmp_path)).ingest(since=since)
    skipped = CalendarEventsCsvAdapter(path=str(tmp_path)).ingest(entity_types=["task"])

    assert [unit.title for unit in result.units] == ["All Day", "Untitled calendar event"]
    assert result.units[0].metadata["all_day"] is True
    assert result.units[1].metadata["description"] == "Missing title"
    assert "start_at" not in result.units[1].metadata
    assert skipped.units == []
    assert get_adapter("calendar_events_csv", path=str(tmp_path)).name == "calendar_events_csv"


def test_calendar_events_csv_source_id_is_deterministic(tmp_path):
    export = tmp_path / "events.csv"
    _write_csv(export, [{"ID": "evt-1", "Title": "Event", "Start": "2026-05-01"}])

    first = CalendarEventsCsvAdapter(path=str(export)).ingest().units[0]
    second = CalendarEventsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
