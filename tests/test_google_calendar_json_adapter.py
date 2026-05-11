from __future__ import annotations

import json

from graph.adapters.google_calendar_json import GoogleCalendarJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


def test_google_calendar_json_imports_top_level_items(tmp_path):
    path = tmp_path / "calendar.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "event-1",
                        "summary": "Design review",
                        "description": "Review the import flows",
                        "htmlLink": "https://calendar.google.com/event?eid=1",
                        "location": "Room 1",
                        "start": {"dateTime": "2025-01-02T03:04:05+00:00", "timeZone": "UTC"},
                        "end": {"dateTime": "2025-01-02T04:04:05+00:00", "timeZone": "UTC"},
                        "attendees": [{"email": "a@example.com", "responseStatus": "accepted"}],
                        "status": "confirmed",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = GoogleCalendarJsonAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == SourceProject.GOOGLE_CALENDAR_JSON
    assert unit.title == "Design review"
    assert "Review the import flows" in unit.content
    assert unit.metadata["source_url"] == "https://calendar.google.com/event?eid=1"
    assert unit.metadata["location"] == "Room 1"
    assert unit.metadata["start"] == {"dateTime": "2025-01-02T03:04:05+00:00", "timeZone": "UTC"}
    assert unit.metadata["end"] == {"dateTime": "2025-01-02T04:04:05+00:00", "timeZone": "UTC"}
    assert unit.metadata["attendees"] == [{"email": "a@example.com", "displayName": None, "responseStatus": "accepted"}]
    assert unit.metadata["status"] == "confirmed"


def test_google_calendar_json_preserves_all_day_dates_and_bare_arrays(tmp_path):
    path = tmp_path / "calendar.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "all-day",
                    "summary": "Away",
                    "start": {"date": "2025-01-02"},
                    "end": {"date": "2025-01-03"},
                    "status": "cancelled",
                }
            ]
        ),
        encoding="utf-8",
    )

    unit = GoogleCalendarJsonAdapter(path=str(path)).ingest().units[0]

    assert unit.metadata["start"] == {"date": "2025-01-02"}
    assert unit.metadata["end"] == {"date": "2025-01-03"}
    assert unit.metadata["status"] == "cancelled"


def test_google_calendar_json_adapter_is_registered():
    assert isinstance(get_adapter("google_calendar_json", path="/tmp/calendar.json"), GoogleCalendarJsonAdapter)
