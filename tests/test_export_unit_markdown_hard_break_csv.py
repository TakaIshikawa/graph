from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_hard_break_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_markdown_hard_break_csv_distinguishes_styles():
    text = export_unit_markdown_hard_break_csv([{"id": "u1", "title": "One", "content": "Two spaces  \nBackslash\\\nPlain"}])

    assert [(row["break_style"], row["line"], row["context"]) for row in rows(text)] == [
        ("two_spaces", "1", "Two spaces"),
        ("backslash", "2", "Backslash"),
    ]


def test_unit_markdown_hard_break_csv_ignores_blank_lines():
    text = export_unit_markdown_hard_break_csv([{"id": "u1", "title": "One", "content": "  \n\\\nText  "}])

    assert [(row["line"], row["context"]) for row in rows(text)] == [("3", "Text")]


def test_unit_markdown_hard_break_csv_path_mode(tmp_path):
    units = [{"id": "u1", "title": "One", "content": "Text  "}]
    path = tmp_path / "hard-breaks.csv"

    expected = export_unit_markdown_hard_break_csv(units)
    stats = export_unit_markdown_hard_break_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
