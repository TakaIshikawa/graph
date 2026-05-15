from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from io import StringIO

from graph.export import export_unit_calendar_event_inventory_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="event",
        title=f"Title {unit_id}",
        content="Content",
        metadata=metadata or {},
        tags=[],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_calendar_event_inventory_csv_empty_input_returns_header():
    assert export_unit_calendar_event_inventory_csv([]) == (
        "unit_id,title,source_project,source_entity_type,start_date,start_datetime,end_date,end_datetime,"
        "due_date,event_date,duration_minutes,attendee_count,location,calendar_id\n"
    )


def test_export_unit_calendar_event_inventory_csv_omits_units_without_event_metadata():
    text = export_unit_calendar_event_inventory_csv([unit("plain", metadata={"status": "todo"})])

    assert rows(text) == []


def test_export_unit_calendar_event_inventory_csv_parses_dates_and_duration():
    text = export_unit_calendar_event_inventory_csv(
        [
            unit(
                "a",
                metadata={
                    "starts_at": "2026-01-01T10:00:00Z",
                    "ends_at": "2026-01-01T11:30:00+00:00",
                    "due_date": date(2026, 1, 2),
                    "event_date": "2026-01-03",
                },
            )
        ]
    )

    row = rows(text)[0]
    assert row["start_date"] == "2026-01-01"
    assert row["start_datetime"] == "2026-01-01T10:00:00+00:00"
    assert row["duration_minutes"] == "90"
    assert row["due_date"] == "2026-01-02"
    assert row["event_date"] == "2026-01-03"


def test_export_unit_calendar_event_inventory_csv_handles_attendees_and_invalid_duration():
    text = export_unit_calendar_event_inventory_csv(
        [
            unit("list", metadata={"start": "not-a-date", "end": "2026-01-01", "attendees": ["a", "", "b"]}),
            unit("string", metadata={"location": " Room 1 ", "calendar_id": "cal", "attendees": "a@example.com; b@example.com"}),
        ]
    )

    result = {row["unit_id"]: row for row in rows(text)}
    assert result["list"]["duration_minutes"] == ""
    assert result["list"]["attendee_count"] == "2"
    assert result["string"]["attendee_count"] == "2"
    assert result["string"]["location"] == "Room 1"
    assert result["string"]["calendar_id"] == "cal"


def test_export_unit_calendar_event_inventory_csv_sorts_by_start_date_then_unit_id():
    text = export_unit_calendar_event_inventory_csv(
        [
            unit("z", metadata={"start": "2026-02-01"}),
            unit("b", metadata={"location": "No date"}),
            unit("a", metadata={"start_at": "2026-01-01"}),
        ]
    )

    assert [row["unit_id"] for row in rows(text)] == ["a", "z", "b"]


def test_export_unit_calendar_event_inventory_csv_path_mode(tmp_path):
    path = tmp_path / "events.csv"
    stats = export_unit_calendar_event_inventory_csv([unit("a", metadata={"event_date": "2026-01-01"})], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["event_date"] == "2026-01-01"
    assert stats["unit_count"] == 1
    assert stats["event_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
