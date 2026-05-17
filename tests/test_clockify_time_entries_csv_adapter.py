from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.clockify_time_entries_csv import ClockifyTimeEntriesCsvAdapter
from graph.types.enums import EdgeRelation
from graph.types.models import SyncState


def test_clockify_time_entries_csv_ingests_normalized_time_entries(tmp_path):
    export = tmp_path / "time_entries.csv"
    export.write_text(
        "Entry ID,Project,Client,Task,Description,User,Start Date,Start Time,End Date,End Time,Duration,Billable,Tags,Hourly Rate\n"
        "entry-1,Graph,Acme,Import,Build Clockify adapter,Taka,2026-05-01,09:00 AM,2026-05-01,10:30 AM,01:30:00,Yes,\"dev, csv\",$120.50\n",
        encoding="utf-8",
    )

    result = ClockifyTimeEntriesCsvAdapter(path=str(export)).ingest(entity_types=["time_entry"])

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "clockify_time_entries_csv"
    assert unit.source_entity_type == "time_entry"
    assert unit.source_id == ClockifyTimeEntriesCsvAdapter(path=str(export)).ingest(entity_types=["time_entry"]).units[0].source_id
    assert unit.metadata["entry_id"] == "entry-1"
    assert unit.metadata["project"] == "Graph"
    assert unit.metadata["client"] == "Acme"
    assert unit.metadata["task"] == "Import"
    assert unit.metadata["description"] == "Build Clockify adapter"
    assert unit.metadata["user"] == "Taka"
    assert unit.metadata["start_at"] == "2026-05-01T09:00:00+00:00"
    assert unit.metadata["end_at"] == "2026-05-01T10:30:00+00:00"
    assert unit.metadata["duration_seconds"] == 5400
    assert unit.metadata["billable"] is True
    assert unit.metadata["tags"] == ["dev", "csv"]
    assert unit.metadata["hourly_rate"] == 120.5
    assert unit.created_at == datetime(2026, 5, 1, 9, tzinfo=timezone.utc)


def test_clockify_time_entries_csv_supports_directory_sparse_rows_and_fallback_ids(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "nested"
    second.mkdir()
    (second / "second.csv").write_text(
        "Project,Task,Start Date,Start Time,End Date,End Time,Billable\n"
        "Graph,Review,2026-05-02,14:00,2026-05-02,14:45,No\n",
        encoding="utf-8",
    )
    first.write_text("Project,Description\n, \nGraph,Planning\n", encoding="utf-8")

    result = ClockifyTimeEntriesCsvAdapter(path=str(tmp_path)).ingest(entity_types=["time_entry"])
    entries = sorted(result.units, key=lambda unit: unit.title)

    assert len(entries) == 2
    assert entries[0].metadata["description"] == "Planning"
    assert entries[1].metadata["duration_seconds"] == 2700
    assert entries[1].metadata["billable"] is False
    again = sorted(ClockifyTimeEntriesCsvAdapter(path=str(tmp_path)).ingest(entity_types=["time_entry"]).units, key=lambda unit: unit.title)
    assert [unit.source_id for unit in entries] == [unit.source_id for unit in again]


def test_clockify_time_entries_csv_filters_since_and_entity_types(tmp_path):
    export = tmp_path / "time_entries.csv"
    export.write_text(
        "Entry ID,Project,Client,Start Date,Start Time,End Date,End Time,Duration\n"
        "old,Old Project,Old Client,2026-04-01,09:00,2026-04-01,10:00,1 hour\n"
        "new,New Project,New Client,2026-05-03,09:00,2026-05-03,11:00,2 hours\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="clockify_time_entries_csv",
        source_entity_type="time_entry",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = ClockifyTimeEntriesCsvAdapter(path=str(export)).ingest(since=since, entity_types=["time_entry", "project", "client"])

    assert {unit.title for unit in result.units if unit.source_entity_type == "time_entry"} == {"New Project"}
    assert {unit.title for unit in result.units if unit.source_entity_type == "project"} == {"New Project"}
    assert {unit.title for unit in result.units if unit.source_entity_type == "client"} == {"New Client"}
    assert ClockifyTimeEntriesCsvAdapter(path=str(export)).ingest(entity_types=["workspace"]).units == []


def test_clockify_time_entries_csv_emits_project_and_client_aggregates_and_edges(tmp_path):
    export = tmp_path / "time_entries.csv"
    export.write_text(
        "Entry ID,Project,Client,Task,User,Start Date,Start Time,End Date,End Time,Duration,Billable,Tags\n"
        "one,Graph,Acme,Import,Ada,2026-05-01,09:00,2026-05-01,10:00,1 hour,Yes,dev\n"
        "two,Graph,Acme,Review,Bob,2026-05-02,09:00,2026-05-02,09:30,30 min,No,review\n"
        "three,Ops,Beta,Support,Ada,2026-05-03,09:00,2026-05-03,10:00,1 hour,Yes,ops\n",
        encoding="utf-8",
    )

    result = ClockifyTimeEntriesCsvAdapter(path=str(export)).ingest()
    projects = {unit.title: unit for unit in result.units if unit.source_entity_type == "project"}
    clients = {unit.title: unit for unit in result.units if unit.source_entity_type == "client"}

    assert set(projects) == {"Graph", "Ops"}
    assert projects["Graph"].metadata["time_entry_count"] == 2
    assert projects["Graph"].metadata["total_duration_seconds"] == 5400
    assert projects["Graph"].metadata["billable_duration_seconds"] == 3600
    assert projects["Graph"].metadata["tasks"] == ["Import", "Review"]
    assert projects["Graph"].metadata["users"] == ["Ada", "Bob"]
    assert clients["Acme"].metadata["projects"] == ["Graph"]
    assert len([edge for edge in result.edges if edge.relation == EdgeRelation.CONTAINS]) == 3
    assert len([edge for edge in result.edges if edge.relation == EdgeRelation.RELATES_TO]) == 3

    project_only = ClockifyTimeEntriesCsvAdapter(path=str(export)).ingest(entity_types=["project"])
    assert {unit.source_entity_type for unit in project_only.units} == {"project"}
    assert project_only.edges == []


def test_clockify_time_entries_csv_handles_empty_missing_and_malformed_files(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    malformed = tmp_path / "malformed.csv"
    malformed.write_bytes(b"\xff\xfe\x00")

    assert ClockifyTimeEntriesCsvAdapter(path=str(empty)).ingest().units == []
    assert ClockifyTimeEntriesCsvAdapter(path=str(tmp_path / "missing.csv")).ingest().units == []
    assert ClockifyTimeEntriesCsvAdapter(path=str(malformed)).ingest().units == []
