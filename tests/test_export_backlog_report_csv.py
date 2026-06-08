from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_backlog_report_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_backlog_report_csv_rolls_up_task_backlog_by_source_and_priority():
    result = rows(
        export_backlog_report_csv(
            [
                {"id": "overdue", "source_project": "Tasks", "metadata": {"status": "open", "priority": "High", "due_date": "2026-06-01"}},
                {"id": "today", "source_project": "Tasks", "metadata": {"status": "pending", "priority": "High", "due": "2026-06-08"}},
                {"id": "future", "source_project": "Tasks", "metadata": {"status": "open", "priority": "low", "deadline": "2026-06-10"}},
                {"id": "done", "source_project": "Tasks", "metadata": {"completed": True, "priority": "High", "due_date": "2026-06-01"}},
                {"id": "none", "source_project": "Tasks", "source_entity_type": "task", "metadata": {"priority": "low", "blocked": True}},
                {"id": "note", "source_project": "Notes", "metadata": {"topic": "planning"}},
            ],
            reference_date="2026-06-08",
        )
    )

    assert result == [
        {
            "source_project": "Tasks",
            "priority": "high",
            "total_tasks": "3",
            "open_tasks": "2",
            "overdue_tasks": "1",
            "due_today_tasks": "1",
            "upcoming_tasks": "0",
            "no_due_date_tasks": "0",
            "blocked_tasks": "0",
            "completed_tasks": "1",
            "high_priority_open_tasks": "2",
            "oldest_due_date": "2026-06-01",
        },
        {
            "source_project": "Tasks",
            "priority": "low",
            "total_tasks": "2",
            "open_tasks": "2",
            "overdue_tasks": "0",
            "due_today_tasks": "0",
            "upcoming_tasks": "1",
            "no_due_date_tasks": "1",
            "blocked_tasks": "1",
            "completed_tasks": "0",
            "high_priority_open_tasks": "0",
            "oldest_due_date": "2026-06-10",
        },
    ]


def test_backlog_report_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "backlog.csv"
    units = [{"id": "a", "tags": ["backlog"], "metadata": {"priority": "P1"}}]

    expected = export_backlog_report_csv(units, reference_date="2026-06-08")
    stats = export_backlog_report_csv(units, path, reference_date="2026-06-08")

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {"path": str(path), "unit_count": 1, "rows_exported": 1, "bytes_written": path.stat().st_size}
