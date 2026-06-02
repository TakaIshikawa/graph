from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_table_headers_to_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_table_header_csv_exports_only_valid_tables_and_escaped_pipes():
    text = export_unit_markdown_table_headers_to_csv(
        [
            {"id": "b", "title": "Beta", "content": "| Not | Table |\n| x | y |\n```md\n| Fake | Header |\n| --- | --- |\n```\n| Name\\|Alias |  |\n| --- | --- |"},
            {"id": "a", "title": "Alpha", "content": "plain"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "b", "title": "Beta", "line_number": "7", "column_index": "1", "header_text": "Name|Alias", "is_empty": "false"},
        {"unit_id": "b", "title": "Beta", "line_number": "7", "column_index": "2", "header_text": "", "is_empty": "true"},
    ]


def test_table_header_csv_path_mode(tmp_path):
    path = tmp_path / "headers.csv"

    stats = export_unit_markdown_table_headers_to_csv([{"id": "u", "content": "| A | B |\n| --- | --- |"}], path)

    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 2
    assert stats["bytes_written"] == path.stat().st_size
