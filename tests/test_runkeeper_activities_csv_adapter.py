from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.runkeeper_activities_csv import RunkeeperActivitiesCsvAdapter
from graph.types.enums import EdgeRelation
from graph.types.models import SyncState


def test_runkeeper_activities_csv_ingests_normalized_activities(tmp_path):
    export = tmp_path / "activities.csv"
    export.write_text(
        "Activity Id,Type,Date,Distance (mi),Duration,Average Pace,Calories Burned,Climb (ft),Average Heart Rate,Notes,GPX File,Route Name\n"
        "rk-1,Running,2026-05-01 07:30:00,3.5,00:32:15,9:13,410,250,145,Morning run,run.gpx,River Loop\n",
        encoding="utf-8",
    )

    result = RunkeeperActivitiesCsvAdapter(path=str(export)).ingest(entity_types=["activity"])

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "runkeeper_activities_csv"
    assert unit.source_entity_type == "activity"
    assert unit.source_id == RunkeeperActivitiesCsvAdapter(path=str(export)).ingest(entity_types=["activity"]).units[0].source_id
    assert unit.metadata["activity_id"] == "rk-1"
    assert unit.metadata["activity_type"] == "Running"
    assert unit.metadata["date"] == "2026-05-01T07:30:00+00:00"
    assert unit.metadata["distance"] == 3.5
    assert unit.metadata["distance_unit"] == "mi"
    assert unit.metadata["duration_seconds"] == 1935
    assert unit.metadata["average_pace"] == "9:13"
    assert unit.metadata["average_pace_seconds"] == 553
    assert unit.metadata["calories"] == 410.0
    assert unit.metadata["climb"] == 250.0
    assert unit.metadata["climb_unit"] == "ft"
    assert unit.metadata["average_heart_rate"] == 145.0
    assert unit.metadata["notes"] == "Morning run"
    assert unit.metadata["gpx_file"] == "run.gpx"
    assert unit.metadata["route_name"] == "River Loop"
    assert unit.created_at == datetime(2026, 5, 1, 7, 30, tzinfo=timezone.utc)


def test_runkeeper_activities_csv_supports_directory_sparse_rows_and_fallback_ids(tmp_path):
    first = tmp_path / "first.csv"
    nested = tmp_path / "nested"
    nested.mkdir()
    first.write_text("Type,Date,Distance\nRunning,2026-05-01,5.0\n,,\n", encoding="utf-8")
    (nested / "second.csv").write_text("Activity Type,Activity Date,Route Name\nWalking,2026-05-02,Park\n", encoding="utf-8")

    result = RunkeeperActivitiesCsvAdapter(path=str(tmp_path)).ingest(entity_types=["activity"])
    units = sorted(result.units, key=lambda unit: unit.created_at)

    assert len(units) == 2
    assert units[0].metadata["activity_type"] == "Running"
    assert units[0].metadata["distance"] == 5.0
    assert units[1].metadata["activity_type"] == "Walking"
    assert units[1].metadata["route_name"] == "Park"
    again = sorted(RunkeeperActivitiesCsvAdapter(path=str(tmp_path)).ingest(entity_types=["activity"]).units, key=lambda unit: unit.created_at)
    assert [unit.source_id for unit in units] == [unit.source_id for unit in again]


def test_runkeeper_activities_csv_filters_since_and_entity_types(tmp_path):
    export = tmp_path / "activities.csv"
    export.write_text(
        "Activity Id,Type,Date,Distance,Route Name\n"
        "old,Running,2026-04-01,2.0,Old Loop\n"
        "new,Cycling,2026-05-03,10.0,New Loop\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="runkeeper_activities_csv",
        source_entity_type="activity",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = RunkeeperActivitiesCsvAdapter(path=str(export)).ingest(since=since, entity_types=["activity", "route", "activity_type"])

    assert {unit.metadata["activity_id"] for unit in result.units if unit.source_entity_type == "activity"} == {"new"}
    assert {unit.title for unit in result.units if unit.source_entity_type == "route"} == {"New Loop"}
    assert {unit.title for unit in result.units if unit.source_entity_type == "activity_type"} == {"Cycling"}
    assert RunkeeperActivitiesCsvAdapter(path=str(export)).ingest(entity_types=["goal"]).units == []


def test_runkeeper_activities_csv_emits_route_and_activity_type_aggregates_and_edges(tmp_path):
    export = tmp_path / "activities.csv"
    export.write_text(
        "Activity Id,Type,Date,Distance,Duration,Calories Burned,Climb,Route Name\n"
        "one,Running,2026-05-01,3.0,30 min,300,100,River Loop\n"
        "two,Running,2026-05-02,4.0,40 min,400,120,River Loop\n"
        "three,Walking,2026-05-03,2.0,35 min,150,50,Park\n",
        encoding="utf-8",
    )

    result = RunkeeperActivitiesCsvAdapter(path=str(export)).ingest()
    routes = {unit.title: unit for unit in result.units if unit.source_entity_type == "route"}
    activity_types = {unit.title: unit for unit in result.units if unit.source_entity_type == "activity_type"}

    assert routes["River Loop"].metadata["activity_count"] == 2
    assert routes["River Loop"].metadata["total_distance"] == 7.0
    assert routes["River Loop"].metadata["total_duration_seconds"] == 4200
    assert routes["River Loop"].metadata["total_calories"] == 700.0
    assert routes["River Loop"].metadata["total_climb"] == 220.0
    assert activity_types["Running"].metadata["routes"] == ["River Loop"]
    assert len([edge for edge in result.edges if edge.relation == EdgeRelation.CONTAINS]) == 3
    assert len([edge for edge in result.edges if edge.relation == EdgeRelation.RELATES_TO]) == 3

    route_only = RunkeeperActivitiesCsvAdapter(path=str(export)).ingest(entity_types=["route"])
    assert {unit.source_entity_type for unit in route_only.units} == {"route"}
    assert route_only.edges == []


def test_runkeeper_activities_csv_handles_empty_missing_and_malformed_files(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    malformed = tmp_path / "malformed.csv"
    malformed.write_bytes(b"\xff\xfe\x00")

    assert RunkeeperActivitiesCsvAdapter(path=str(empty)).ingest().units == []
    assert RunkeeperActivitiesCsvAdapter(path=str(tmp_path / "missing.csv")).ingest().units == []
    assert RunkeeperActivitiesCsvAdapter(path=str(malformed)).ingest().units == []


def test_runkeeper_activities_csv_is_registered():
    assert isinstance(get_adapter("runkeeper_activities_csv"), RunkeeperActivitiesCsvAdapter)
