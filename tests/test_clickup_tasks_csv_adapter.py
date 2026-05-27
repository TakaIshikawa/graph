from __future__ import annotations

from graph.adapters.clickup_tasks_csv import ClickUpTasksCsvAdapter
from graph.adapters.registry import get_adapter


def test_clickup_tasks_csv_ingests_task_fields(tmp_path):
    export = tmp_path / "tasks.csv"
    export.write_text(
        "Task ID,Task Name,Status,Priority,Assignees,Tags,Date Created,Date Updated,Due Date,URL,Description\nT1,Ship report,complete,high,\"Ada, Grace\",\"ops, weekly\",2024-01-01T00:00:00Z,2024-01-02T00:00:00Z,2024-01-05,https://clickup.com/t/T1,Done\n",
        encoding="utf-8",
    )

    unit = ClickUpTasksCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.metadata["task_id"] == "T1"
    assert unit.metadata["status"] == "complete"
    assert unit.metadata["priority"] == "high"
    assert unit.metadata["assignees"] == ["Ada", "Grace"]
    assert unit.metadata["tags"] == ["ops", "weekly"]
    assert unit.metadata["due_date"] == "2024-01-05"
    assert isinstance(get_adapter("clickup_tasks_csv"), ClickUpTasksCsvAdapter)


def test_clickup_tasks_csv_uses_stable_fallback_and_open_status(tmp_path):
    export = tmp_path / "tasks.csv"
    export.write_text("Task Name,Status\nOpen task,open\n", encoding="utf-8")

    first = ClickUpTasksCsvAdapter(path=str(export)).ingest().units[0]
    second = ClickUpTasksCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.metadata["status"] == "open"
