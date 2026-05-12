from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.peloton_workout_summary_csv import PelotonWorkoutSummaryCsvAdapter
from graph.types.models import SyncState


def test_peloton_workout_summary_csv_ingests_normalized_metrics(tmp_path):
    export = tmp_path / "summary.csv"
    export.write_text(
        "Workout ID,Start Time,Class Title,Instructor,Discipline,Duration,Output,Distance,Calories,Leaderboard Rank,Class URL\n"
        "wo-1,2026-05-01T09:00:00Z,Intervals,Ada,Cycling,30 min,250.5,10.25 mi,310,\"1,234\",https://peloton.example/classes/1\n",
        encoding="utf-8",
    )

    result = PelotonWorkoutSummaryCsvAdapter(path=str(export)).ingest()

    unit = result.units[0]
    assert unit.source_project == "peloton_workout_summary_csv"
    assert unit.source_entity_type == "workout"
    assert unit.source_id == PelotonWorkoutSummaryCsvAdapter(path=str(export)).ingest().units[0].source_id
    assert unit.metadata["workout_id"] == "wo-1"
    assert unit.metadata["duration_seconds"] == 1800
    assert unit.metadata["output"] == 250.5
    assert unit.metadata["distance"] == 10.25
    assert unit.metadata["calories"] == 310
    assert unit.metadata["leaderboard_rank"] == 1234
    assert unit.metadata["start_time"] == "2026-05-01T09:00:00+00:00"
    assert unit.created_at == datetime(2026, 5, 1, 9, tzinfo=timezone.utc)


def test_peloton_workout_summary_csv_filters_since_and_entity_types(tmp_path):
    export = tmp_path / "summary.csv"
    export.write_text(
        "Start Time,Class Title,Discipline\n"
        "2026-04-01T09:00:00Z,Old Ride,Cycling\n"
        "2026-05-03T09:00:00Z,New Ride,Cycling\n",
        encoding="utf-8",
    )

    since = SyncState(source_project="peloton_workout_summary_csv", source_entity_type="workout", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
    result = PelotonWorkoutSummaryCsvAdapter(path=str(export)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["New Ride"]
    assert PelotonWorkoutSummaryCsvAdapter(path=str(export)).ingest(entity_types=["class"]).units == []


def test_peloton_workout_summary_csv_fallback_source_id_is_stable(tmp_path):
    export = tmp_path / "summary.csv"
    export.write_text(
        "Start Time,Class Title\n2026-05-01T09:00:00Z,Stable Ride\n",
        encoding="utf-8",
    )

    first = PelotonWorkoutSummaryCsvAdapter(path=str(export)).ingest().units[0]
    second = PelotonWorkoutSummaryCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("peloton_workout_summary_csv:")
