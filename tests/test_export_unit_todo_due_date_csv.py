from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_todo_due_date_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_todo_due_date_csv_extracts_valid_task_markers():
    text = export_units_to_todo_due_date_csv(
        [{"id": "a", "title": "Alpha", "content": "- [ ] call due:2026-06-01\n- [x] ship @due(2026-06-02)\n- [ ] bad due:2026-99-99\nplain due:2026-06-03"}]
    )

    assert rows(text) == [
        {"unit_id": "a", "title": "Alpha", "line_number": "1", "task_text": "call due:2026-06-01", "completed": "false", "due_date": "2026-06-01", "marker_style": "due_colon"},
        {"unit_id": "a", "title": "Alpha", "line_number": "2", "task_text": "ship @due(2026-06-02)", "completed": "true", "due_date": "2026-06-02", "marker_style": "at_due"},
    ]


def test_unit_todo_due_date_csv_supports_dataview_marker():
    assert rows(export_units_to_todo_due_date_csv([{"id": "a", "content": "- [ ] Task [due:: 2026-06-03]"}]))[0]["marker_style"] == "due_dataview"
