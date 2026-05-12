from __future__ import annotations

from graph.adapters.garmin_activities_csv import GarminActivitiesCsvAdapter
from graph.adapters.registry import get_adapter


def test_garmin_activities_csv_ingests_metrics_and_tolerates_bad_optional_values(tmp_path):
    export = tmp_path / "garmin.csv"
    export.write_text("Activity Type,Date,Title,Distance,Calories,Time,Avg HR,Max HR,Total Ascent,Activity ID\nRunning,2026-05-01,Run,5.2 km,abc,00:30:00,150,bad,100,gar-1\n", encoding="utf-8")

    result = GarminActivitiesCsvAdapter(path=str(export)).ingest()

    unit = next(unit for unit in result.units if unit.source_entity_type == "activity")
    assert unit.source_id == "garmin_activities_csv:gar-1"
    assert unit.metadata["distance"] == 5.2
    assert unit.metadata["duration_seconds"] == 1800
    assert unit.metadata["avg_hr"] == 150
    assert "max_hr" not in unit.metadata
    assert "Running" in unit.tags
    assert get_adapter("garmin_activities_csv", path=str(export)).name == "garmin_activities_csv"


def test_garmin_activities_csv_emits_activity_type_aggregates(tmp_path):
    export = tmp_path / "garmin.csv"
    export.write_text(
        "\n".join(
            [
                "Activity Type,Date,Title,Distance,Time,Activity ID",
                "Running,2026-05-01,Run 1,5.0,00:30:00,gar-1",
                " running ,2026-05-02,Run 2,7.0,00:45:00,gar-2",
                ",2026-05-03,Untyped,2.0,00:10:00,gar-3",
            ]
        ),
        encoding="utf-8",
    )

    result = GarminActivitiesCsvAdapter(path=str(export)).ingest()

    aggregates = [unit for unit in result.units if unit.source_entity_type == "activity_type"]
    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate.metadata["normalized_activity_type"] == "running"
    assert aggregate.metadata["activity_count"] == 2
    assert aggregate.metadata["total_distance"] == 12.0
    assert aggregate.metadata["total_duration_seconds"] == 4500
    assert len(result.edges) == 2
    assert {edge.to_unit_id for edge in result.edges} == {aggregate.source_id}
