from __future__ import annotations

from graph.adapters.todoist_tasks_csv import TodoistTasksCsvAdapter


def test_todoist_tasks_csv_ingests_task_metadata(tmp_path):
    export = tmp_path / "todoist.csv"
    export.write_text(
        "ID,Content,Project,Section,Labels,Priority,Due Date,Completed,Created At,Completed At,Comments\n"
        "1,Ship report,Work,Reports,\"Ops,Writing\",4,2024-02-01,true,2024-01-01,2024-01-05,Done\n",
        encoding="utf-8",
    )

    unit = TodoistTasksCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Ship report"
    assert unit.metadata["project"] == "Work"
    assert unit.metadata["completed"] is True
    assert unit.metadata["status"] == "completed"
    assert unit.tags == ["ops", "writing"]


def test_todoist_tasks_csv_active_task_and_stable_id(tmp_path):
    export = tmp_path / "todoist.csv"
    export.write_text("Content,Project\nOpen task,Inbox\n", encoding="utf-8")

    first = TodoistTasksCsvAdapter(path=str(export)).ingest().units[0]
    second = TodoistTasksCsvAdapter(path=str(export)).ingest().units[0]

    assert first.metadata["status"] == "active"
    assert first.source_id == second.source_id
