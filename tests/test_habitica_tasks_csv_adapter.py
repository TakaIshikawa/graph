from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.habitica_tasks_csv import HabiticaTasksCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_habitica_tasks_csv_ingests_task_metadata_and_tags(tmp_path):
    export = tmp_path / "habitica.csv"
    export.write_text(
        "Task ID,Type,Text,Notes,Tags,Priority,Difficulty,Value,Due Date,Created At,Updated At,Completed At,Status,Checklist\n"
        "task-1,todo,Write adapter,Include tests,\"work,coding\",1.5,hard,10,2026-05-10,2026-05-01T09:00:00Z,2026-05-02T10:00:00Z,,open,\"draft; test\"\n"
        "task-2,daily,Review streak,,routine,1,easy,-2,,2026-05-03,2026-05-04,2026-05-04T12:00:00Z,,\n",
        encoding="utf-8",
    )

    result = HabiticaTasksCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    unit = result.units[0]
    assert unit.source_project == "habitica_tasks_csv"
    assert unit.source_id == "habitica_tasks_csv:task-1"
    assert unit.source_entity_type == "task"
    assert unit.content_type == ContentType.METADATA
    assert unit.title == "Write adapter"
    assert unit.metadata["type"] == "todo"
    assert unit.metadata["notes"] == "Include tests"
    assert unit.metadata["tags"] == ["work", "coding"]
    assert unit.metadata["priority"] == 1.5
    assert unit.metadata["difficulty"] == "hard"
    assert unit.metadata["value"] == 10.0
    assert unit.metadata["due_date"] == "2026-05-10T00:00:00+00:00"
    assert unit.metadata["created_at"] == "2026-05-01T09:00:00+00:00"
    assert unit.metadata["updated_at"] == "2026-05-02T10:00:00+00:00"
    assert unit.metadata["status"] == "open"
    assert unit.metadata["checklist"] == "draft; test"
    assert {"habitica", "todo", "open", "work", "coding"}.issubset(set(unit.tags))
    assert result.units[1].metadata["status"] == "completed"


def test_habitica_tasks_csv_skips_blank_rows_sorts_and_filters_since(tmp_path):
    export = tmp_path / "habitica.csv"
    export.write_text(
        "Task ID,Type,Text,Created At,Updated At\n"
        ",,,,\n"
        "old,todo,Old,2026-05-01,2026-05-01\n"
        "new,habit,New,2026-05-01,2026-05-03\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="habitica_tasks_csv", source_entity_type="task", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = HabiticaTasksCsvAdapter(path=str(export)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == ["habitica_tasks_csv:new"]
    assert HabiticaTasksCsvAdapter(path=str(export)).ingest(entity_types=["transaction"]).units == []
    all_units = HabiticaTasksCsvAdapter(path=str(export)).ingest().units
    assert [(unit.created_at, unit.source_id) for unit in all_units] == sorted((unit.created_at, unit.source_id) for unit in all_units)


def test_habitica_tasks_csv_source_ids_are_stable_without_task_ids(tmp_path):
    export = tmp_path / "habitica.csv"
    export.write_text(
        "Type,Text,Notes,Due Date,Created At\n"
        "reward,Buy coffee,Use coins,,2026-05-05\n",
        encoding="utf-8",
    )

    first = HabiticaTasksCsvAdapter(path=str(export)).ingest().units[0]
    second = HabiticaTasksCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("habitica_tasks_csv:")
