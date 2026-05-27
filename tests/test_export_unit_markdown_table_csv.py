from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_table_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_no_tables_has_header():
    assert export_units_to_markdown_table_csv([{"id": "u", "content": "No table"}]) == "unit_id,title,heading,start_line,column_count,row_count,header_cells\n"


def test_pipe_table_with_alignment_row():
    text = export_units_to_markdown_table_csv([{"id": "u", "title": "Unit", "content": "# H\n| A | B |\n| :- | -: |\n| 1 | 2 |"}])
    assert rows(text) == [{"unit_id": "u", "title": "Unit", "heading": "H", "start_line": "2", "column_count": "2", "row_count": "1", "header_cells": "A | B"}]


def test_multiple_tables_stable_by_unit_and_line():
    text = export_units_to_markdown_table_csv([
        {"id": "b", "content": "| C | D |\n| - | - |"},
        {"id": "a", "content": "## One\n| A | B |\n| - | - |\n| 1 | 2 |\n\n## Two\n| X | Y | Z |\n| - | - | - |"},
    ])
    assert [row["unit_id"] + ":" + row["start_line"] for row in rows(text)] == ["a:2", "a:7", "b:1"]
    assert rows(text)[0]["row_count"] == "1"
    assert rows(text)[1]["heading"] == "Two"
