from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.google_fit_daily_csv import GoogleFitDailyCsvAdapter
from graph.types.enums import EdgeRelation
from graph.types.models import SyncState


def test_google_fit_daily_csv_ingests_normalized_daily_activity(tmp_path):
    export = tmp_path / "daily.csv"
    export.write_text(
        "Date,Steps,Distance (km),Move Minutes,Heart Points,Calories,Active Calories,Average Heart Rate,Min Heart Rate,Max Heart Rate,Sleep Duration,Weight (kg),Source\n"
        "2026-05-01,12000,8.5,45,30,2300,700,72.5,55,145,7:30:00,70.2,Pixel Watch\n",
        encoding="utf-8",
    )

    result = GoogleFitDailyCsvAdapter(path=str(export)).ingest(entity_types=["daily_activity"])

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "google_fit_daily_csv"
    assert unit.source_entity_type == "daily_activity"
    assert unit.source_id == GoogleFitDailyCsvAdapter(path=str(export)).ingest(entity_types=["daily_activity"]).units[0].source_id
    assert unit.metadata["date"] == "2026-05-01"
    assert unit.metadata["steps"] == 12000
    assert unit.metadata["distance"] == 8.5
    assert unit.metadata["distance_unit"] == "km"
    assert unit.metadata["move_minutes"] == 45
    assert unit.metadata["heart_points"] == 30
    assert unit.metadata["calories"] == 2300.0
    assert unit.metadata["active_calories"] == 700.0
    assert unit.metadata["average_heart_rate"] == 72.5
    assert unit.metadata["min_heart_rate"] == 55.0
    assert unit.metadata["max_heart_rate"] == 145.0
    assert unit.metadata["sleep_duration_seconds"] == 27000
    assert unit.metadata["weight"] == 70.2
    assert unit.metadata["weight_unit"] == "kg"
    assert unit.metadata["source"] == "Pixel Watch"
    assert unit.created_at == datetime(2026, 5, 1, tzinfo=timezone.utc)


def test_google_fit_daily_csv_supports_directory_and_sparse_rows(tmp_path):
    first = tmp_path / "first.csv"
    nested = tmp_path / "nested"
    nested.mkdir()
    first.write_text("Date,Steps,Calories\n2026-05-01,1000,\n,\n", encoding="utf-8")
    (nested / "second.csv").write_text("Activity Date,Move Min,Source\n2026-05-02,20,Phone\n", encoding="utf-8")

    result = GoogleFitDailyCsvAdapter(path=str(tmp_path)).ingest(entity_types=["daily_activity"])
    units = sorted(result.units, key=lambda unit: unit.metadata["date"])

    assert [unit.metadata["date"] for unit in units] == ["2026-05-01", "2026-05-02"]
    assert units[0].metadata["steps"] == 1000
    assert units[1].metadata["move_minutes"] == 20
    assert units[1].metadata["source"] == "Phone"


def test_google_fit_daily_csv_filters_since_and_entity_types(tmp_path):
    export = tmp_path / "daily.csv"
    export.write_text(
        "Date,Steps,Calories\n"
        "2026-04-01,1000,1800\n"
        "2026-05-03,2000,1900\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="google_fit_daily_csv",
        source_entity_type="daily_activity",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = GoogleFitDailyCsvAdapter(path=str(export)).ingest(since=since, entity_types=["daily_activity", "metric"])

    assert {unit.metadata["date"] for unit in result.units if unit.source_entity_type == "daily_activity"} == {"2026-05-03"}
    metrics = {unit.metadata["metric"]: unit for unit in result.units if unit.source_entity_type == "metric"}
    assert set(metrics) == {"steps", "calories"}
    assert metrics["steps"].metadata["total"] == 2000.0
    assert GoogleFitDailyCsvAdapter(path=str(export)).ingest(entity_types=["workout"]).units == []


def test_google_fit_daily_csv_emits_metric_aggregates_and_edges(tmp_path):
    export = tmp_path / "daily.csv"
    export.write_text(
        "Date,Steps,Distance,Move Minutes,Calories\n"
        "2026-05-01,1000,1.2,10,1800\n"
        "2026-05-02,3000,3.4,,2000\n",
        encoding="utf-8",
    )

    result = GoogleFitDailyCsvAdapter(path=str(export)).ingest()
    metrics = {unit.metadata["metric"]: unit for unit in result.units if unit.source_entity_type == "metric"}

    assert set(metrics) == {"steps", "distance", "move_minutes", "calories"}
    assert metrics["steps"].metadata["daily_count"] == 2
    assert metrics["steps"].metadata["total"] == 4000.0
    assert metrics["steps"].metadata["minimum"] == 1000.0
    assert metrics["steps"].metadata["maximum"] == 3000.0
    assert metrics["steps"].metadata["average"] == 2000.0
    assert metrics["move_minutes"].metadata["daily_count"] == 1
    assert len([edge for edge in result.edges if edge.relation == EdgeRelation.CONTAINS]) == 7

    metric_only = GoogleFitDailyCsvAdapter(path=str(export)).ingest(entity_types=["metric"])
    assert {unit.source_entity_type for unit in metric_only.units} == {"metric"}
    assert metric_only.edges == []


def test_google_fit_daily_csv_handles_empty_missing_and_malformed_files(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    malformed = tmp_path / "malformed.csv"
    malformed.write_bytes(b"\xff\xfe\x00")

    assert GoogleFitDailyCsvAdapter(path=str(empty)).ingest().units == []
    assert GoogleFitDailyCsvAdapter(path=str(tmp_path / "missing.csv")).ingest().units == []
    assert GoogleFitDailyCsvAdapter(path=str(malformed)).ingest().units == []
