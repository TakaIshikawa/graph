from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.sleep_as_android_csv import SleepAsAndroidCsvAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_sleep_as_android_csv_ingests_sleep_session(tmp_path):
    export = tmp_path / "sleep.csv"
    _write_csv(
        export,
        [
            {
                "Sleep Start": "2026-05-01 23:00",
                "Sleep End": "2026-05-02 07:15",
                "Duration": "8:15:00",
                "Quality": "86%",
                "Deep Sleep %": "42%",
                "Snoring Time": "12",
                "Noise Level": "25.5",
                "Tags": "travel;late",
                "Comments": "Slept well",
            }
        ],
    )

    result = SleepAsAndroidCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.SLEEP_AS_ANDROID_CSV
    assert unit.source_entity_type == "sleep_session"
    assert unit.metadata["sleep_start"] == "2026-05-01T23:00:00+00:00"
    assert unit.metadata["sleep_end"] == "2026-05-02T07:15:00+00:00"
    assert unit.metadata["duration_seconds"] == 29700
    assert unit.metadata["quality"] == 86.0
    assert unit.metadata["deep_sleep"] == 42.0
    assert unit.metadata["snoring"] == 12.0
    assert unit.metadata["noise"] == 25.5
    assert unit.metadata["tags"] == ["travel", "late"]
    assert unit.updated_at == datetime(2026, 5, 1, 23, tzinfo=timezone.utc)
    assert "Tags: travel, late" in unit.content


def test_sleep_as_android_csv_directory_since_entity_filter_sparse_and_invalid_numbers(tmp_path):
    _write_csv(
        tmp_path / "one.csv",
        [
            {"Start": "2026-04-30 23:00", "End": "2026-05-01 07:00", "Rating": "80"},
            {"From": "2026-05-03 22:30", "To": "2026-05-04 06:30", "Length": "480 min", "DeepSleep": "bad"},
            {"From": "", "Duration": ""},
        ],
    )
    _write_csv(tmp_path / "two.csv", [{"Start Time": "not a date", "Duration": "7 hours", "Avg Noise": "quiet"}])
    since = SyncState(
        source_project="sleep_as_android_csv",
        source_entity_type="sleep_session",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = SleepAsAndroidCsvAdapter(path=str(tmp_path)).ingest(since=since)
    skipped = SleepAsAndroidCsvAdapter(path=str(tmp_path)).ingest(entity_types=["sleep"])

    by_duration = {unit.metadata["duration_seconds"]: unit for unit in result.units}
    assert set(by_duration) == {25200, 28800}
    assert "deep_sleep" not in by_duration[28800].metadata
    assert "noise" not in by_duration[25200].metadata
    assert skipped.units == []
    assert get_adapter("sleep_as_android_csv", path=str(tmp_path)).name == "sleep_as_android_csv"


def test_sleep_as_android_csv_source_id_is_deterministic(tmp_path):
    export = tmp_path / "sleep.csv"
    _write_csv(export, [{"Start": "2026-05-01", "Duration": "8 hours", "Comment": "same"}])

    first = SleepAsAndroidCsvAdapter(path=str(export)).ingest().units[0]
    second = SleepAsAndroidCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
