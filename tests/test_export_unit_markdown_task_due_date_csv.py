from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_task_due_date_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_markdown_task_due_dates_and_cleans_text():
    result = rows(export_units_to_markdown_task_due_date_csv([{"id": "u", "title": "Tasks", "content": "- [ ] write tests due:2026-06-02\n- [x] ship [due:: 2026-06-03]\n- [ ] ignored"}]))

    assert [(row["line_number"], row["checked"], row["due_date"], row["task_text"]) for row in result] == [
        ("1", "false", "2026-06-02", "write tests"),
        ("2", "true", "2026-06-03", "ship"),
    ]
