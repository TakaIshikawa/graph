from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.ticktick_tasks_csv import TickTickTasksCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_ticktick_tasks_csv_ingests_common_columns(tmp_path):
    export = tmp_path / "ticktick.csv"
    export.write_text(
        "ID,Title,Content,List Name,Tags,Priority,Status,Created Time,Modified Time,Due Date,Completed Time,Timezone\n"
        "task-1,Write adapter,Map columns,Inbox,\"work,coding\",High,Normal,2026-05-01T09:00:00Z,2026-05-02T10:00:00Z,2026-05-03,,America/New_York\n"
        "task-2,Ship tests,,Personal,home,1,Completed,2026-05-04,2026-05-05,2026-05-06,2026-05-07T08:30:00Z,UTC\n",
        encoding="utf-8",
    )

    result = TickTickTasksCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    unit = result.units[0]
    assert unit.source_project == "ticktick_tasks_csv"
    assert unit.source_id == "ticktick_tasks_csv:task-1"
    assert unit.source_entity_type == "task"
    assert unit.content_type == ContentType.METADATA
    assert unit.title == "Write adapter"
    assert unit.content.startswith("Write adapter\nMap columns")
    assert unit.metadata["list_name"] == "Inbox"
    assert unit.metadata["priority"] == "high"
    assert unit.metadata["status"] == "normal"
    assert unit.metadata["tags"] == ["work", "coding"]
    assert unit.metadata["created_at"] == "2026-05-01T09:00:00+00:00"
    assert unit.metadata["modified_at"] == "2026-05-02T10:00:00+00:00"
    assert unit.metadata["due_at"] == "2026-05-03T00:00:00+00:00"
    assert unit.metadata["timezone"] == "America/New_York"
    assert unit.metadata["source_file"] == "ticktick.csv"
    assert unit.metadata["source_row"]["Title"] == "Write adapter"
    assert {"ticktick", "task", "Inbox", "normal", "high", "work", "coding"}.issubset(set(unit.tags))
    assert result.units[1].metadata["completed_at"] == "2026-05-07T08:30:00+00:00"


def test_ticktick_tasks_csv_directory_skips_bad_files_sorts_and_filters_since(tmp_path):
    old = tmp_path / "old.csv"
    old.write_text("ID,Title,Created Time,Modified Time\nold,Old,2026-05-01,2026-05-01\n", encoding="utf-8")
    new = tmp_path / "new.csv"
    new.write_text(
        "ID,Title,List Name,Created Time,Modified Time\nnew,New,Today,2026-05-01,2026-05-03\n",
        encoding="utf-8",
    )
    ignored = tmp_path / "notes.txt"
    ignored.write_text("ID,Title\nignored,Ignored\n", encoding="utf-8")
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\xff\xfe\x00")
    blank = tmp_path / "blank.csv"
    blank.write_text("ID,Title\n,\n", encoding="utf-8")
    since = SyncState(source_project="ticktick_tasks_csv", source_entity_type="task", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = TickTickTasksCsvAdapter(path=str(tmp_path)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == ["ticktick_tasks_csv:new"]
    assert TickTickTasksCsvAdapter(path=str(tmp_path)).ingest(entity_types=["transaction"]).units == []
    all_units = TickTickTasksCsvAdapter(path=str(tmp_path)).ingest().units
    assert [unit.source_id for unit in all_units] == ["ticktick_tasks_csv:old", "ticktick_tasks_csv:new"]
    assert [(unit.updated_at, unit.source_id) for unit in all_units] == sorted((unit.updated_at, unit.source_id) for unit in all_units)


def test_ticktick_tasks_csv_source_ids_are_stable_without_id(tmp_path):
    export = tmp_path / "ticktick.csv"
    export.write_text(
        "Title,Content,List Name,Due Date,Created Time\n"
        "Buy milk,Whole milk,Errands,2026-05-05,2026-05-01\n",
        encoding="utf-8",
    )

    first = TickTickTasksCsvAdapter(path=str(export)).ingest().units[0]
    second = TickTickTasksCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("ticktick_tasks_csv:")
