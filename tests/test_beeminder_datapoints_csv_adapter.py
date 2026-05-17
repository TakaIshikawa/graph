from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.beeminder_datapoints_csv import BeeminderDatapointsCsvAdapter
from graph.types.enums import EdgeRelation
from graph.types.models import SyncState


def test_beeminder_datapoints_csv_ingests_normalized_datapoints(tmp_path):
    export = tmp_path / "datapoints.csv"
    export.write_text(
        "Datapoint Id,Goal,Goal Slug,Date,Timestamp,Value,Comment,Request ID,Updated At,Created At,Daystamp,Tags\n"
        "dp-1,Write,write,2026-05-01,2026-05-01 08:30:00,750,words,req-1,2026-05-01 09:00:00,2026-05-01 08:00:00,20260501,\"draft, morning\"\n",
        encoding="utf-8",
    )

    result = BeeminderDatapointsCsvAdapter(path=str(export)).ingest(entity_types=["datapoint"])

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "beeminder_datapoints_csv"
    assert unit.source_entity_type == "datapoint"
    assert unit.source_id == BeeminderDatapointsCsvAdapter(path=str(export)).ingest(entity_types=["datapoint"]).units[0].source_id
    assert unit.metadata["datapoint_id"] == "dp-1"
    assert unit.metadata["goal"] == "Write"
    assert unit.metadata["goal_slug"] == "write"
    assert unit.metadata["date"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["timestamp"] == "2026-05-01T08:30:00+00:00"
    assert unit.metadata["value"] == 750.0
    assert unit.metadata["comment"] == "words"
    assert unit.metadata["request_id"] == "req-1"
    assert unit.metadata["updated_at"] == "2026-05-01T09:00:00+00:00"
    assert unit.metadata["created_at"] == "2026-05-01T08:00:00+00:00"
    assert unit.metadata["daystamp"] == "20260501"
    assert unit.metadata["tags"] == ["draft", "morning"]
    assert unit.created_at == datetime(2026, 5, 1, 8, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 5, 1, 9, tzinfo=timezone.utc)


def test_beeminder_datapoints_csv_supports_directory_sparse_rows_and_fallback_ids(tmp_path):
    first = tmp_path / "first.csv"
    nested = tmp_path / "nested"
    nested.mkdir()
    first.write_text("Goal,Date,Value,Comment\nWrite,2026-05-01,500,words\n,,,\n", encoding="utf-8")
    (nested / "second.csv").write_text("Goal Slug,Daystamp,Value\nrun,20260502,3.1\n", encoding="utf-8")

    result = BeeminderDatapointsCsvAdapter(path=str(tmp_path)).ingest(entity_types=["datapoint"])
    units = sorted(result.units, key=lambda unit: unit.title.casefold())

    assert len(units) == 2
    assert units[0].metadata["goal_slug"] == "run"
    assert units[0].metadata["value"] == 3.1
    assert units[1].metadata["goal"] == "Write"
    assert units[1].metadata["comment"] == "words"
    again = sorted(
        BeeminderDatapointsCsvAdapter(path=str(tmp_path)).ingest(entity_types=["datapoint"]).units,
        key=lambda unit: unit.title.casefold(),
    )
    assert [unit.source_id for unit in units] == [unit.source_id for unit in again]


def test_beeminder_datapoints_csv_filters_since_and_entity_types(tmp_path):
    export = tmp_path / "datapoints.csv"
    export.write_text(
        "Datapoint Id,Goal,Goal Slug,Date,Value,Updated At\n"
        "old,Write,write,2026-04-01,100,2026-04-01\n"
        "new,Run,run,2026-05-03,5,2026-05-03\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="beeminder_datapoints_csv",
        source_entity_type="datapoint",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = BeeminderDatapointsCsvAdapter(path=str(export)).ingest(since=since, entity_types=["datapoint", "goal"])

    assert {unit.metadata["datapoint_id"] for unit in result.units if unit.source_entity_type == "datapoint"} == {"new"}
    assert {unit.title for unit in result.units if unit.source_entity_type == "goal"} == {"Run"}
    assert BeeminderDatapointsCsvAdapter(path=str(export)).ingest(entity_types=["user"]).units == []


def test_beeminder_datapoints_csv_emits_goal_aggregates_and_edges(tmp_path):
    export = tmp_path / "datapoints.csv"
    export.write_text(
        "Datapoint Id,Goal,Goal Slug,Date,Value,Comment,Request ID,Tags\n"
        "one,Write,write,2026-05-01,500,draft,req-1,words\n"
        "two,Write,write,2026-05-02,750,edit,req-2,words\n"
        "three,Run,run,2026-05-03,3.1,miles,req-3,fitness\n",
        encoding="utf-8",
    )

    result = BeeminderDatapointsCsvAdapter(path=str(export)).ingest()
    goals = {unit.title: unit for unit in result.units if unit.source_entity_type == "goal"}

    assert goals["Write"].metadata["datapoint_count"] == 2
    assert goals["Write"].metadata["value_total"] == 1250.0
    assert goals["Write"].metadata["value_minimum"] == 500.0
    assert goals["Write"].metadata["value_maximum"] == 750.0
    assert goals["Write"].metadata["first_datapoint_at"] == "2026-05-01T00:00:00+00:00"
    assert goals["Write"].metadata["last_datapoint_at"] == "2026-05-02T00:00:00+00:00"
    assert goals["Write"].metadata["request_ids"] == ["req-1", "req-2"]
    assert len([edge for edge in result.edges if edge.relation == EdgeRelation.CONTAINS]) == 3

    goal_only = BeeminderDatapointsCsvAdapter(path=str(export)).ingest(entity_types=["goal"])
    assert {unit.source_entity_type for unit in goal_only.units} == {"goal"}
    assert goal_only.edges == []


def test_beeminder_datapoints_csv_handles_empty_missing_and_malformed_files(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    malformed = tmp_path / "malformed.csv"
    malformed.write_bytes(b"\xff\xfe\x00")

    assert BeeminderDatapointsCsvAdapter(path=str(empty)).ingest().units == []
    assert BeeminderDatapointsCsvAdapter(path=str(tmp_path / "missing.csv")).ingest().units == []
    assert BeeminderDatapointsCsvAdapter(path=str(malformed)).ingest().units == []
