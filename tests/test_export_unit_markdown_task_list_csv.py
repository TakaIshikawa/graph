from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_task_list_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_task_list_csv_exports_checked_state_depth_and_heading():
    text = export_units_to_markdown_task_list_csv(
        [{"id": "u", "title": "Tasks", "source_path": "tasks.md", "content": "# Today\n- [ ] Open item\n  - [x] Done item"}]
    )

    assert _rows(text) == [
        {
            "unit_id": "u",
            "title": "Tasks",
            "source_path": "tasks.md",
            "source": "",
            "line_number": "2",
            "checked": "false",
            "marker": "-",
            "nesting_depth": "0",
            "task_text": "Open item",
            "parent_heading": "Today",
        },
        {
            "unit_id": "u",
            "title": "Tasks",
            "source_path": "tasks.md",
            "source": "",
            "line_number": "3",
            "checked": "true",
            "marker": "-",
            "nesting_depth": "1",
            "task_text": "Done item",
            "parent_heading": "Today",
        },
    ]


def test_task_list_csv_ignores_fenced_markers():
    text = export_units_to_markdown_task_list_csv([{"id": "u", "content": "```\n- [x] skip\n```\n- [ ] keep"}])

    assert [(row["checked"], row["task_text"], row["line_number"]) for row in _rows(text)] == [("false", "keep", "4")]
