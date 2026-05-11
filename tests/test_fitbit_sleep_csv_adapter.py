from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.fitbit_sleep_csv import FitbitSleepCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_fitbit_sleep_csv_ingests_score_and_stage_exports(tmp_path):
    export = tmp_path / "sleep.csv"
    _write_csv(
        export,
        [
            {
                "Start Time": "2025-01-01T22:30:00Z",
                "End Time": "2025-01-02T06:45:00Z",
                "Minutes Asleep": "430",
                "Minutes Awake": "35",
                "Number of Awakenings": "12",
                "Time in Bed": "495",
                "Sleep Score": "86",
                "Deep Sleep": "75",
                "Light Sleep": "250",
                "REM Sleep": "105",
            }
        ],
    )

    result = FitbitSleepCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.FITBIT_SLEEP_CSV
    assert unit.metadata["start_time"] == "2025-01-01T22:30:00+00:00"
    assert unit.metadata["end_time"] == "2025-01-02T06:45:00+00:00"
    assert unit.metadata["minutes_asleep"] == 430
    assert unit.metadata["minutes_awake"] == 35
    assert unit.metadata["awakenings"] == 12
    assert unit.metadata["time_in_bed"] == 495
    assert unit.metadata["sleep_score"] == 86
    assert unit.metadata["deep_minutes"] == 75
    assert "good_sleep" in unit.tags


def test_fitbit_sleep_csv_missing_optional_metrics_and_filters(tmp_path):
    export = tmp_path / "sleep.csv"
    _write_csv(
        export,
        [
            {"Date": "01/01/2025", "Minutes Asleep": "400"},
            {"Date": "02/01/2025", "Minutes Asleep": "420"},
        ],
    )
    since = SyncState(
        source_project="fitbit_sleep_csv",
        source_entity_type="sleep",
        last_sync_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
    )

    result = FitbitSleepCsvAdapter(path=str(export)).ingest(since=since)

    assert len(result.units) == 1
    assert result.units[0].metadata["minutes_asleep"] == 420
    assert result.units[0].metadata["sleep_score"] is None
    assert FitbitSleepCsvAdapter(path=str(export)).ingest(entity_types=["activity"]).units == []


def test_fitbit_sleep_csv_directory_and_registry(tmp_path):
    _write_csv(tmp_path / "one.csv", [{"Date": "2025-01-01", "Minutes Asleep": "400"}])
    _write_csv(tmp_path / "two.csv", [{"Date": "2025-01-02", "Minutes Asleep": "410"}])

    result = FitbitSleepCsvAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    assert get_adapter("fitbit_sleep_csv", path=str(tmp_path)).name == "fitbit_sleep_csv"
