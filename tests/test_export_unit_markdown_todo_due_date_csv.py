from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_todo_due_date_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_supported_markdown_todo_due_date_markers():
    text = export_units_to_markdown_todo_due_date_csv(
        [
            {
                "id": "u1",
                "title": "Tasks",
                "content": "- [ ] draft due: 2026-06-10\n- [x] ship 📅 2026-06-11\n- [ ] review [due:: 2026-06-12]",
            }
        ]
    )

    assert rows(text) == [
        {"unit_id": "u1", "title": "Tasks", "line_number": "1", "task_text": "draft due: 2026-06-10", "due_date": "2026-06-10", "completed": "false"},
        {"unit_id": "u1", "title": "Tasks", "line_number": "2", "task_text": "ship 📅 2026-06-11", "due_date": "2026-06-11", "completed": "true"},
        {"unit_id": "u1", "title": "Tasks", "line_number": "3", "task_text": "review [due:: 2026-06-12]", "due_date": "2026-06-12", "completed": "false"},
    ]


def test_ignores_invalid_dates_and_non_task_lines():
    text = export_units_to_markdown_todo_due_date_csv(
        [
            {
                "id": "u1",
                "content": "- [ ] impossible due: 2026-99-99\nplain due: 2026-06-10\n- [ ] valid due: 2026-06-11",
            }
        ]
    )

    assert rows(text) == [
        {"unit_id": "u1", "title": "", "line_number": "3", "task_text": "valid due: 2026-06-11", "due_date": "2026-06-11", "completed": "false"}
    ]


def test_markdown_todo_due_date_export_has_deterministic_output():
    text = export_units_to_markdown_todo_due_date_csv(
        [
            {"id": "b", "content": "- [ ] later due: 2026-06-20"},
            {"id": "a", "metadata": {"title": "Alpha"}, "content": "Intro\n- [x] first [due:: 2026-06-01]"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "a", "title": "Alpha", "line_number": "2", "task_text": "first [due:: 2026-06-01]", "due_date": "2026-06-01", "completed": "true"},
        {"unit_id": "b", "title": "", "line_number": "1", "task_text": "later due: 2026-06-20", "due_date": "2026-06-20", "completed": "false"},
    ]
