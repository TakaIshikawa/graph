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
    result = PelotonWorkoutSummaryCsvAdapter(path=str(export)).ingest(since=since, entity_types=["workout"])

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


def test_peloton_workout_summary_csv_emits_month_aggregates_and_edges(tmp_path):
    export = tmp_path / "summary.csv"
    export.write_text(
        "Workout ID,Start Time,Class Title,Instructor,Discipline,Duration,Output,Distance,Calories\n"
        "wo-1,2026-05-01T09:00:00Z,Ride,Ada,Cycling,30 min,0,10,0\n"
        "wo-2,2026-05-03T09:00:00Z,Run,Bob,Running,20 min,,3,200\n"
        "wo-3,2026-06-01T09:00:00Z,Yoga,Ada,Yoga,,,,\n",
        encoding="utf-8",
    )

    result = PelotonWorkoutSummaryCsvAdapter(path=str(export)).ingest(entity_types=["workout", "workout_month"])

    assert PelotonWorkoutSummaryCsvAdapter(path=str(export)).entity_types == ["workout", "workout_month", "instructor"]
    months = sorted((unit for unit in result.units if unit.source_entity_type == "workout_month"), key=lambda unit: unit.metadata["month"])
    assert [month.metadata["month"] for month in months] == ["2026-05", "2026-06"]
    may = months[0]
    may_workouts = [unit for unit in result.units if unit.source_entity_type == "workout" and unit.created_at.month == 5]
    assert may.metadata["workout_count"] == 2
    assert may.metadata["total_duration_seconds"] == 3000
    assert may.metadata["total_output"] == 0.0
    assert may.metadata["total_distance"] == 13.0
    assert may.metadata["total_calories"] == 200
    assert may.metadata["disciplines"] == ["Cycling", "Running"]
    assert may.metadata["instructors"] == ["Ada", "Bob"]
    assert may.metadata["workout_source_ids"] == sorted(unit.source_id for unit in may_workouts)
    assert len(result.edges) == 3

    month_only = PelotonWorkoutSummaryCsvAdapter(path=str(export)).ingest(entity_types=["workout_month"])
    assert {unit.source_entity_type for unit in month_only.units} == {"workout_month"}
    assert month_only.edges == []


def test_peloton_workout_summary_csv_emits_instructor_aggregates_and_edges(tmp_path):
    export = tmp_path / "summary.csv"
    export.write_text(
        "Workout ID,Start Time,Class Title,Instructor,Discipline,Duration,Output,Distance,Calories\n"
        "wo-1,2026-05-01T09:00:00Z,Ride,Ada,Cycling,30 min,250,10,300\n"
        "wo-2,2026-05-03T09:00:00Z,Run,ada,Running,20 min,,3,\n"
        "wo-3,2026-06-01T09:00:00Z,Yoga,Bob,Yoga,,,,100\n",
        encoding="utf-8",
    )

    result = PelotonWorkoutSummaryCsvAdapter(path=str(export)).ingest(entity_types=["workout", "instructor"])
    instructors = {unit.title: unit for unit in result.units if unit.source_entity_type == "instructor"}

    assert set(instructors) == {"Ada", "Bob"}
    ada = instructors["Ada"]
    assert ada.metadata["workout_count"] == 2
    assert ada.metadata["disciplines"] == ["Cycling", "Running"]
    assert ada.metadata["total_duration_seconds"] == 3000
    assert ada.metadata["total_output"] == 250.0
    assert ada.metadata["total_distance"] == 13.0
    assert ada.metadata["total_calories"] == 300
    assert ada.metadata["first_workout_at"] == "2026-05-01T09:00:00+00:00"
    assert ada.metadata["last_workout_at"] == "2026-05-03T09:00:00+00:00"
    assert ada.metadata["source_files"] == ["summary.csv"]
    assert len([edge for edge in result.edges if edge.metadata["relation_type"] == "workout_instructor"]) == 3

    instructor_only = PelotonWorkoutSummaryCsvAdapter(path=str(export)).ingest(entity_types=["instructor"])
    assert {unit.source_entity_type for unit in instructor_only.units} == {"instructor"}
    assert instructor_only.edges == []
