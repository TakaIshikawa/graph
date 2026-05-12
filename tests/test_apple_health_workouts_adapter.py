from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.apple_health_workouts import AppleHealthWorkoutsAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_apple_health_workouts_ingests_workout_records(tmp_path):
    export = tmp_path / "export.xml"
    export.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_US">
  <Workout workoutActivityType="HKWorkoutActivityTypeRunning"
    duration="31.5" durationUnit="min"
    totalDistance="5.25" totalDistanceUnit="km"
    totalEnergyBurned="412" totalEnergyBurnedUnit="kcal"
    sourceName="Fitness" sourceVersion="10.0"
    device="&lt;&lt;HKDevice: 0x1&gt;, name:Apple Watch&gt;"
    creationDate="2025-01-02 07:31:00 +0900"
    startDate="2025-01-02 07:00:00 +0900"
    endDate="2025-01-02 07:31:30 +0900">
    <MetadataEntry key="HKMetadataKeyIndoorWorkout" value="0"/>
    <MetadataEntry key="HKMetadataKeyExternalUUID" value="workout-abc-123"/>
    <WorkoutRoute sourceName="Fitness" sourceVersion="10.0" creationDate="2025-01-02 07:32:00 +0900"/>
  </Workout>
  <Record type="HKQuantityTypeIdentifierStepCount" value="100"/>
</HealthData>
""",
        encoding="utf-8",
    )

    result = AppleHealthWorkoutsAdapter(path=str(export)).ingest()

    workouts = [unit for unit in result.units if unit.source_entity_type == "workout"]
    assert len(workouts) == 1
    unit = workouts[0]
    assert unit.source_project == SourceProject.APPLE_HEALTH_WORKOUTS
    assert unit.source_entity_type == "workout"
    assert unit.source_id.startswith("apple_health_workouts:")
    assert unit.title == "Running on 2025-01-01"
    assert unit.content_type == ContentType.METADATA
    assert unit.created_at == datetime(2025, 1, 1, 22, 0, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 1, 22, 31, 30, tzinfo=timezone.utc)
    assert unit.tags == ["apple_health", "workout", "running"]
    assert unit.metadata["activity_type"] == "Running"
    assert unit.metadata["workout_activity_type"] == "HKWorkoutActivityTypeRunning"
    assert unit.metadata["duration"] == 31.5
    assert unit.metadata["duration_unit"] == "min"
    assert unit.metadata["distance"] == 5.25
    assert unit.metadata["distance_unit"] == "km"
    assert unit.metadata["calories"] == 412.0
    assert unit.metadata["calories_unit"] == "kcal"
    assert unit.metadata["source_name"] == "Fitness"
    assert unit.metadata["source_version"] == "10.0"
    assert "Apple Watch" in unit.metadata["device"]
    assert unit.metadata["metadata_entries"]["HKMetadataKeyIndoorWorkout"] == "0"
    assert unit.metadata["metadata_entries"]["HKMetadataKeyExternalUUID"] == "workout-abc-123"
    assert unit.metadata["routes"][0]["element"] == "WorkoutRoute"
    assert unit.metadata["routes"][0]["sourceName"] == "Fitness"
    assert unit.metadata["source_file"] == "export.xml"
    assert "Activity: Running" in unit.content
    assert "Distance: 5.25 km" in unit.content


def test_apple_health_workouts_allows_missing_optional_fields(tmp_path):
    export = tmp_path / "export.xml"
    export.write_text(
        """<HealthData>
  <Workout workoutActivityType="HKWorkoutActivityTypeYoga" startDate="2025-03-10 08:00:00 +0000"/>
  <Workout workoutActivityType="HKWorkoutActivityTypeCycling" duration="bad" startDate="not a date"/>
</HealthData>
""",
        encoding="utf-8",
    )

    result = AppleHealthWorkoutsAdapter(path=str(tmp_path)).ingest()

    workouts = [unit for unit in result.units if unit.source_entity_type == "workout"]
    assert len(workouts) == 1
    unit = workouts[0]
    assert unit.title == "Yoga on 2025-03-10"
    assert unit.metadata["duration"] is None
    assert unit.metadata["distance"] is None
    assert unit.metadata["calories"] is None
    assert unit.metadata["source_name"] == ""
    assert unit.metadata["metadata_entries"] == {}
    assert unit.metadata["routes"] == []


def test_apple_health_workouts_filters_by_sync_state(tmp_path):
    export = tmp_path / "export.xml"
    export.write_text(
        """<HealthData>
  <Workout workoutActivityType="HKWorkoutActivityTypeWalking" startDate="2025-01-01 00:00:00 +0000"/>
  <Workout workoutActivityType="HKWorkoutActivityTypeSwimming" startDate="2025-01-02 00:00:00 +0000"/>
</HealthData>
""",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="apple_health_workouts",
        source_entity_type="workout",
        last_sync_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
    )

    result = AppleHealthWorkoutsAdapter(path=str(export)).ingest(since=since)

    workouts = [unit for unit in result.units if unit.source_entity_type == "workout"]
    assert len(workouts) == 1
    assert workouts[0].metadata["activity_type"] == "Swimming"


def test_apple_health_workouts_entity_type_filtering(tmp_path):
    export = tmp_path / "export.xml"
    export.write_text(
        """<HealthData>
  <Workout workoutActivityType="HKWorkoutActivityTypeWalking" startDate="2025-01-01 00:00:00 +0000"/>
</HealthData>
""",
        encoding="utf-8",
    )

    result = AppleHealthWorkoutsAdapter(path=str(export)).ingest(entity_types=["record"])

    assert result.units == []
    assert result.edges == []


def test_apple_health_workouts_source_id_is_stable(tmp_path):
    export = tmp_path / "export.xml"
    export.write_text(
        """<HealthData>
  <Workout workoutActivityType="HKWorkoutActivityTypeWalking"
    duration="20" sourceName="Fitness" startDate="2025-01-01 00:00:00 +0000"/>
</HealthData>
""",
        encoding="utf-8",
    )

    first = [unit for unit in AppleHealthWorkoutsAdapter(path=str(export)).ingest().units if unit.source_entity_type == "workout"][0]
    second = [unit for unit in AppleHealthWorkoutsAdapter(path=str(export)).ingest().units if unit.source_entity_type == "workout"][0]

    assert first.source_id == second.source_id


def test_apple_health_workouts_emits_monthly_aggregates(tmp_path):
    export = tmp_path / "export.xml"
    export.write_text(
        """<HealthData>
  <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="30" startDate="2025-01-01 00:00:00 +0000"/>
  <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="45" startDate="2025-01-20 00:00:00 +0000"/>
  <Workout workoutActivityType="HKWorkoutActivityTypeWalking" duration="10" startDate="2025-01-20 00:00:00 +0000"/>
  <Workout workoutActivityType="HKWorkoutActivityTypeRunning" duration="20" startDate="2025-02-01 00:00:00 +0000"/>
</HealthData>
""",
        encoding="utf-8",
    )

    result = AppleHealthWorkoutsAdapter(path=str(export)).ingest()

    aggregates = [unit for unit in result.units if unit.source_entity_type == "monthly_aggregate"]
    assert len(aggregates) == 3
    january_running = next(unit for unit in aggregates if unit.metadata["month"] == "2025-01" and unit.metadata["activity_type"] == "Running")
    assert january_running.metadata["workout_count"] == 2
    assert january_running.metadata["total_duration"] == 75.0
    assert len([edge for edge in result.edges if edge.to_unit_id == january_running.source_id]) == 2


def test_apple_health_workouts_adapter_is_registered():
    assert "apple_health_workouts" in list_adapters()
    adapter = get_adapter("apple_health_workouts", path="/tmp/export.xml")
    assert adapter.name == "apple_health_workouts"
