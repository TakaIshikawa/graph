from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.google_fit_activity_csv import GoogleFitActivityCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.models import SyncState


def test_google_fit_activity_csv_ingests_activity_rows(tmp_path):
    export = tmp_path / "fit.csv"
    export.write_text(
        "Start Time,End Time,Activity Type,Steps,Distance (km),Calories (kcal),Duration\n"
        "2026-05-01T08:00:00Z,2026-05-01T08:45:00Z,Walking,4200,3.2,180,45 min\n",
        encoding="utf-8",
    )

    unit = GoogleFitActivityCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "google_fit_activity_csv"
    assert unit.source_entity_type == "activity"
    assert unit.metadata["start_time"] == "2026-05-01T08:00:00+00:00"
    assert unit.metadata["end_time"] == "2026-05-01T08:45:00+00:00"
    assert unit.metadata["activity_type"] == "Walking"
    assert unit.metadata["steps"] == 4200
    assert unit.metadata["distance"] == 3.2
    assert unit.metadata["distance_unit"] == "km"
    assert unit.metadata["calories"] == 180.0
    assert unit.metadata["duration_seconds"] == 2700
    assert unit.metadata["source_file"] == "fit.csv"
    assert unit.metadata["source_row"] == 2
    assert {"google_fit", "activity", "Walking"}.issubset(set(unit.tags))


def test_google_fit_activity_csv_accepts_column_variants_skips_bad_and_filters(tmp_path):
    (tmp_path / "old.csv").write_text("Start,End,Type,Step Count,Distance,Calories Burned,Duration Seconds\n2026-05-01,2026-05-01,Run,10,1,2,3\n", encoding="utf-8")
    (tmp_path / "new.csv").write_text("Begin Time,Finish Time,Exercise,Steps,Distance (m),Energy,Elapsed Time\n2026-05-03T00:00:00Z,2026-05-03T00:10:00Z,Cycling,0,1000,50,00:10:00\n,,,,,,\n", encoding="utf-8")
    (tmp_path / "bad.csv").write_text('"unterminated', encoding="utf-8")
    since = SyncState(source_project="google_fit_activity_csv", source_entity_type="activity", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = GoogleFitActivityCsvAdapter(path=str(tmp_path)).ingest(since=since)

    assert [unit.metadata["activity_type"] for unit in result.units] == ["Cycling"]
    assert result.units[0].metadata["duration_seconds"] == 600
    assert GoogleFitActivityCsvAdapter(path=str(tmp_path)).ingest(entity_types=["daily_activity"]).units == []


def test_google_fit_activity_csv_is_registered():
    assert "google_fit_activity_csv" in list_adapters()
    assert isinstance(get_adapter("google-fit-activity-csv"), GoogleFitActivityCsvAdapter)
