from __future__ import annotations

from graph.adapters.calendly_events_csv import CalendlyEventsCsvAdapter
from graph.adapters.registry import get_adapter


def test_calendly_events_csv_ingests_scheduled_events(tmp_path):
    export = tmp_path / "calendly.csv"
    export.write_text("Event UUID,Event Type,Invitee Name,Invitee Email,Start Time,End Time,Timezone,Status,Cancellation Reason,Location,Created At,Goal\nc1,Intro,Ada,ada@example.com,2026-05-01T10:00:00Z,2026-05-01T10:30:00Z,UTC,canceled,Conflict,Zoom,2026-04-30T00:00:00Z,Discuss import\n", encoding="utf-8")

    unit = CalendlyEventsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_entity_type == "scheduled_event"
    assert unit.metadata["status"] == "canceled"
    assert unit.metadata["cancellation_reason"] == "Conflict"
    assert unit.metadata["questions"] == {"Goal": "Discuss import"}
    assert "Location: Zoom" in unit.content


def test_calendly_events_csv_is_registered():
    assert isinstance(get_adapter("calendly-events-csv"), CalendlyEventsCsvAdapter)
