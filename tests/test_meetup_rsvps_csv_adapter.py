from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.meetup_rsvps_csv import MeetupRsvpsCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_meetup_rsvps_csv_ingests_event_group_location_and_url_metadata(tmp_path):
    export = tmp_path / "meetup.csv"
    export.write_text(
        "event_id,event_name,group_name,event_date,rsvp_response,venue_name,location,event_url,description,rsvp_date\n"
        "evt-1,Graph Night,Knowledge Builders,2026-05-01T18:30:00Z,yes,Main Hall,1 Main St,https://meetup.example/events/evt-1,Talks and demos,2026-04-20T09:00:00Z\n"
        "evt-2,Coffee Walk,City Walkers,2026-05-03,no,Park Gate,2 Park Ave,https://meetup.example/events/evt-2,,2026-04-21\n",
        encoding="utf-8",
    )

    result = MeetupRsvpsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    unit = result.units[0]
    assert unit.source_project == "meetup_rsvps_csv"
    assert unit.source_id == "meetup_rsvps_csv:evt-1"
    assert unit.source_entity_type == "rsvp"
    assert unit.content_type == ContentType.METADATA
    assert unit.title == "Graph Night"
    assert unit.metadata["event_id"] == "evt-1"
    assert unit.metadata["event_name"] == "Graph Night"
    assert unit.metadata["group_name"] == "Knowledge Builders"
    assert unit.metadata["event_date"] == "2026-05-01T18:30:00+00:00"
    assert unit.metadata["rsvp_response"] == "yes"
    assert unit.metadata["venue_name"] == "Main Hall"
    assert unit.metadata["location"] == "1 Main St"
    assert unit.metadata["event_url"] == "https://meetup.example/events/evt-1"
    assert unit.metadata["description"] == "Talks and demos"
    assert unit.metadata["rsvp_date"] == "2026-04-20T09:00:00+00:00"
    assert {"meetup", "rsvp", "yes", "Knowledge Builders"}.issubset(set(unit.tags))
    assert "Graph Night" in unit.content
    assert "URL: https://meetup.example/events/evt-1" in unit.content


def test_meetup_rsvps_csv_handles_missing_optional_fields_and_filters_since(tmp_path):
    export = tmp_path / "meetup.csv"
    export.write_text(
        "event_name,group_name,event_date,rsvp_response,venue_name,event_url\n"
        ",,,,,\n"
        "Old Event,,2026-05-01,yes,,\n"
        "New Event,Future Group,2026-05-03,waitlist,,\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="meetup_rsvps_csv", source_entity_type="rsvp", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = MeetupRsvpsCsvAdapter(path=str(export)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["New Event"]
    assert "venue_name" not in result.units[0].metadata
    assert "event_url" not in result.units[0].metadata
    assert "None" not in result.units[0].content
    assert MeetupRsvpsCsvAdapter(path=str(export)).ingest(entity_types=["event"]).units == []
    all_units = MeetupRsvpsCsvAdapter(path=str(export)).ingest().units
    assert [(unit.created_at, unit.source_id) for unit in all_units] == sorted((unit.created_at, unit.source_id) for unit in all_units)


def test_meetup_rsvps_csv_source_ids_are_stable_without_event_ids(tmp_path):
    export = tmp_path / "meetup.csv"
    export.write_text(
        "event_name,group_name,event_date,rsvp_response,event_url\n"
        "URL Event,Web Group,2026-05-05,yes,https://meetup.example/events/url-event\n"
        "Digest Event,No URL Group,2026-05-06,maybe,\n",
        encoding="utf-8",
    )

    first = MeetupRsvpsCsvAdapter(path=str(export)).ingest().units
    second = MeetupRsvpsCsvAdapter(path=str(export)).ingest().units

    assert [unit.source_id for unit in first] == [unit.source_id for unit in second]
    assert all(unit.source_id.startswith("meetup_rsvps_csv:") for unit in first)
