from __future__ import annotations

from graph.adapters.garmin_activities_csv import GarminActivitiesCsvAdapter
from graph.adapters.registry import get_adapter


def test_garmin_activities_csv_ingests_metrics_and_tolerates_bad_optional_values(tmp_path):
    export = tmp_path / "garmin.csv"
    export.write_text("Activity Type,Date,Title,Distance,Calories,Time,Avg HR,Max HR,Total Ascent,Activity ID\nRunning,2026-05-01,Run,5.2 km,abc,00:30:00,150,bad,100,gar-1\n", encoding="utf-8")

    result = GarminActivitiesCsvAdapter(path=str(export)).ingest()

    unit = result.units[0]
    assert unit.source_id == "garmin_activities_csv:gar-1"
    assert unit.metadata["distance"] == 5.2
    assert unit.metadata["duration_seconds"] == 1800
    assert unit.metadata["avg_hr"] == 150
    assert "max_hr" not in unit.metadata
    assert "Running" in unit.tags
    assert get_adapter("garmin_activities_csv", path=str(export)).name == "garmin_activities_csv"
