from __future__ import annotations

from graph.adapters.fitbit_daily_activity_csv import FitbitDailyActivityCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject


def test_fitbit_daily_activity_csv_ingests_daily_activity_rows(tmp_path):
    export = tmp_path / "activity.csv"
    export.write_text(
        "Date,Steps,Distance (km),Floors,Calories Burned,Minutes Lightly Active,Minutes Fairly Active,Minutes Very Active,Sedentary Minutes,Activity Calories,Steps Goal,Steps Progress\n"
        "2026-05-01,12000,8.4,12,2400,45,20,15,600,900,10000,120\n",
        encoding="utf-8",
    )

    result = FitbitDailyActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.FITBIT_DAILY_ACTIVITY_CSV
    assert unit.source_entity_type == "daily_activity"
    assert unit.metadata["date"] == "2026-05-01"
    assert unit.metadata["steps"] == 12000
    assert unit.metadata["distance"] == 8.4
    assert unit.metadata["distance_unit"] == "km"
    assert unit.metadata["floors"] == 12
    assert unit.metadata["calories"] == 2400
    assert unit.metadata["lightly_active_minutes"] == 45
    assert unit.metadata["fairly_active_minutes"] == 20
    assert unit.metadata["very_active_minutes"] == 15
    assert unit.metadata["sedentary_minutes"] == 600
    assert unit.metadata["activity_calories"] == 900
    assert unit.metadata["goals"]["steps_goal"] == 10000
    assert unit.metadata["progress"]["steps_progress"] == 120


def test_fitbit_daily_activity_csv_handles_alternate_casing_and_missing_optional_minutes(tmp_path):
    export = tmp_path / "activity.csv"
    export.write_text(
        "activity date,steps,distance,calories out\n"
        "2026-05-02,5000,3.1,1800\n",
        encoding="utf-8",
    )

    result = FitbitDailyActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["date"] == "2026-05-02"
    assert unit.metadata["steps"] == 5000
    assert unit.metadata["distance"] == 3.1
    assert unit.metadata["calories"] == 1800
    assert "very_active_minutes" not in unit.metadata
    assert "goals" not in unit.metadata


def test_fitbit_daily_activity_csv_is_registered():
    assert "fitbit_daily_activity_csv" in list_adapters()
    assert isinstance(get_adapter("fitbit-daily-activity-csv"), FitbitDailyActivityCsvAdapter)
