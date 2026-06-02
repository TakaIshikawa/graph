from __future__ import annotations

from graph.adapters.todoist_completed_csv import TodoistCompletedCsvAdapter
from graph.adapters.registry import get_adapter


def test_todoist_completed_csv_ingests_completed_tasks(tmp_path):
    path = tmp_path / "done.csv"
    path.write_text("Task ID,Content,Project,Section,Labels,Priority,Completed At,Created At,Due Date,URL,Recurring\n42,Ship adapter,Graph,Imports,\"work,urgent\",4,2026-05-02T10:00:00Z,2026-05-01,2026-05-03,https://todoist.test/task/42,false\n", encoding="utf-8")

    unit = TodoistCompletedCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == "todoist_completed_csv"
    assert unit.source_id == "todoist_completed_csv:42"
    assert unit.source_entity_type == "completed_task"
    assert unit.metadata["labels"] == ["work", "urgent"]
    assert unit.metadata["recurring"] is False
    assert unit.tags == ["todoist", "completed", "Graph", "work", "urgent"]
    assert isinstance(get_adapter("todoist_completed_csv"), TodoistCompletedCsvAdapter)
