from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.apple_calendar_events_csv import AppleCalendarEventsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_apple_calendar_events_csv_ingests_event_metadata(tmp_path):
    export = tmp_path / "events.csv"
    _write_csv(
        export,
        [
            {
                "Summary": "Planning",
                "Calendar Name": "Work",
                "Start Date": "2026-05-01 10:00",
                "End Date": "2026-05-01 11:00",
                "All-day": "false",
                "Where": "Room 1",
                "Event URL": "https://calendar.example/event",
                "Notes": "Roadmap discussion",
                "Invitees": "a@example.com; b@example.com",
                "Organiser": "owner@example.com",
                "Repeat": "weekly",
            }
        ],
    )

    result = AppleCalendarEventsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "apple_calendar_events_csv"
    assert unit.source_entity_type == "calendar_event"
    assert unit.title == "Planning"
    assert unit.metadata["title"] == "Planning"
    assert unit.metadata["calendar"] == "Work"
    assert unit.metadata["start_at"] == "2026-05-01T10:00:00+00:00"
    assert unit.metadata["end_at"] == "2026-05-01T11:00:00+00:00"
    assert unit.metadata["all_day"] is False
    assert unit.metadata["location"] == "Room 1"
    assert unit.metadata["url"] == "https://calendar.example/event"
    assert unit.metadata["notes"] == "Roadmap discussion"
    assert unit.metadata["attendees"] == ["a@example.com", "b@example.com"]
    assert unit.metadata["organizer"] == "owner@example.com"
    assert unit.metadata["recurrence"] == "weekly"
    assert unit.metadata["source_file"] == "events.csv"
    assert unit.metadata["source_row"] == 2


def test_apple_calendar_events_csv_directory_filters_bad_files_and_registry(tmp_path):
    _write_csv(
        tmp_path / "events.csv",
        [
            {"UID": "old", "Title": "Old", "Start": "2026-04-30", "End": "2026-04-30"},
            {"UID": "all-day", "Title": "All Day", "Start": "2026-05-03", "End": "2026-05-04", "All Day": "yes"},
            {"UID": "", "Title": "", "Start": "", "End": ""},
        ],
    )
    (tmp_path / "bad.csv").write_bytes(b"\xff\xff")
    since = SyncState(
        source_project="apple_calendar_events_csv",
        source_entity_type="calendar_event",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = AppleCalendarEventsCsvAdapter(path=str(tmp_path)).ingest(since=since)
    skipped = AppleCalendarEventsCsvAdapter(path=str(tmp_path)).ingest(entity_types=["task"])

    assert [unit.title for unit in result.units] == ["All Day"]
    assert result.units[0].metadata["all_day"] is True
    assert skipped.units == []
    assert get_adapter("apple-calendar-events-csv", path=str(tmp_path)).name == "apple_calendar_events_csv"


def test_apple_calendar_events_csv_source_id_is_deterministic_and_deduped(tmp_path):
    _write_csv(tmp_path / "events.csv", [{"ID": "evt-1", "Title": "Event", "Start": "2026-05-01"}])
    _write_csv(tmp_path / "copy.csv", [{"ID": "evt-1", "Title": "Event", "Start": "2026-05-01"}])

    result = AppleCalendarEventsCsvAdapter(path=str(tmp_path)).ingest()
    second = AppleCalendarEventsCsvAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id == second.units[0].source_id
