from __future__ import annotations

from graph.adapters.peloton_workouts_csv import PelotonWorkoutsCsvAdapter
from graph.adapters.registry import get_adapter


def test_peloton_workouts_csv_ingests_metrics_and_tags(tmp_path):
    export = tmp_path / "peloton.csv"
    export.write_text("Workout Timestamp,Fitness Discipline,Title,Instructor Name,Length,Total Output,Calories Burned,Distance,Avg Watts,Workout URL\n2026-05-01T09:00:00Z,Cycling,Intervals,Ada,30 min,250,300,10.5,150,https://example.com/workout\n", encoding="utf-8")

    result = PelotonWorkoutsCsvAdapter(path=str(export)).ingest()

    unit = result.units[0]
    assert unit.metadata["length_seconds"] == 1800
    assert unit.metadata["total_output"] == 250.0
    assert unit.metadata["workout_url"] == "https://example.com/workout"
    assert {"Cycling", "Ada"}.issubset(set(unit.tags))
    assert get_adapter("peloton_workouts_csv", path=str(export)).name == "peloton_workouts_csv"
