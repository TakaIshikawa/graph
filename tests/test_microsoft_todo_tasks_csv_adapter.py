from __future__ import annotations

from graph.adapters.microsoft_todo_tasks_csv import MicrosoftTodoTasksCsvAdapter


def test_microsoft_todo_tasks_csv_represents_completed_and_incomplete_tasks(tmp_path):
    export = tmp_path / "todo.csv"
    export.write_text(
        "Title,List Name,Status,Importance,Due Date,Completed Date,Created Date,Notes,Recurrence\n"
        "Ship patch,Work,Completed,High,2026-05-01,2026-05-02,2026-04-30,Done notes,Weekly\n"
        "Plan trip,Personal,Not Started,Normal,2026-06-01,,2026-05-01,Book train,\n",
        encoding="utf-8",
    )

    units = MicrosoftTodoTasksCsvAdapter(path=str(export)).ingest().units

    by_title = {unit.title: unit for unit in units}
    assert by_title["Ship patch"].metadata["status"] == "completed"
    assert by_title["Plan trip"].metadata["status"] == "incomplete"
    assert by_title["Ship patch"].metadata["due_date"] == "2026-05-01T00:00:00+00:00"
    assert by_title["Ship patch"].metadata["completed_date"] == "2026-05-02T00:00:00+00:00"
    assert by_title["Ship patch"].metadata["created_date"] == "2026-04-30T00:00:00+00:00"
    assert by_title["Ship patch"].metadata["list_name"] == "Work"
    assert by_title["Ship patch"].metadata["importance"] == "High"
    assert by_title["Ship patch"].metadata["recurrence"]["pattern"] == "Weekly"
