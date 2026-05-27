from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_list_structure_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_list_structure_counts_list_kinds_and_depth():
    text = export_units_to_markdown_list_structure_csv(
        [
            {"id": "a", "content": "- one\n  - [ ] task\n1. ordered\nprose - not a list"},
            {"id": "b", "content": "plain\n  2) nested ordered"},
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "unordered_items": "2",
            "ordered_items": "1",
            "task_items": "1",
            "max_indent_level": "1",
            "list_line_count": "3",
        },
        {
            "unit_id": "b",
            "unordered_items": "0",
            "ordered_items": "1",
            "task_items": "0",
            "max_indent_level": "1",
            "list_line_count": "1",
        },
    ]


def test_markdown_list_structure_path_mode(tmp_path):
    path = tmp_path / "lists.csv"
    stats = export_units_to_markdown_list_structure_csv([{"id": "a", "content": "* item"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["list_line_count"] == "1"
    assert stats["rows_exported"] == 1
