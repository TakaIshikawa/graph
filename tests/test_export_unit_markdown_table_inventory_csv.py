from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_table_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_table_inventory_exports_multiple_tables_deterministically():
    text = export_units_to_markdown_table_inventory_csv(
        [
            {"id": "b", "title": "Beta", "content": "| C | D |\n| - | - |\n| 3 | 4 |"},
            {"id": "a", "title": "Alpha", "content": "Intro\n| A | B |\n| :- | -: |\n| 1 | 2 |\n\n| X | Y | Z |\n| - | - | - |"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "a", "title": "Alpha", "table_index": "1", "start_line": "2", "header_column_count": "2", "data_row_count": "1", "header_preview": "A | B"},
        {"unit_id": "a", "title": "Alpha", "table_index": "2", "start_line": "6", "header_column_count": "3", "data_row_count": "0", "header_preview": "X | Y | Z"},
        {"unit_id": "b", "title": "Beta", "table_index": "1", "start_line": "1", "header_column_count": "2", "data_row_count": "1", "header_preview": "C | D"},
    ]


def test_markdown_table_inventory_ignores_malformed_pipe_paragraphs_and_code():
    text = export_units_to_markdown_table_inventory_csv(
        [{"id": "u", "content": "A | B\nnot | separator\n```\n| X | Y |\n| - | - |\n```\n| Kept | Yes |\n| - | - |"}]
    )

    assert len(rows(text)) == 1
    assert rows(text)[0]["header_preview"] == "Kept | Yes"


def test_markdown_table_inventory_empty_input_has_header():
    assert export_units_to_markdown_table_inventory_csv([]) == "unit_id,title,table_index,start_line,header_column_count,data_row_count,header_preview\n"
