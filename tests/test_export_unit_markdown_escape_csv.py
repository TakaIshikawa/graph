from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_escape_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_markdown_escape_csv_detects_markdown_punctuation():
    text = export_unit_markdown_escape_csv([{"id": "u1", "title": "One", "content": r"\*literal\* and \[x\] plus \# tag"}])

    assert [(row["escaped_character"], row["line"]) for row in rows(text)] == [("#", "1"), ("*", "1"), ("*", "1"), ("[", "1"), ("]", "1")]


def test_unit_markdown_escape_csv_ignores_filesystem_backslashes():
    text = export_unit_markdown_escape_csv([{"id": "u1", "title": "One", "content": r"C:\Users\Name\file.txt"}])

    assert rows(text) == []


def test_unit_markdown_escape_csv_is_stable_by_unit_and_line():
    units = [
        {"id": "b", "title": "B", "content": "Line \\#\nNext \\*"},
        {"id": "a", "title": "A", "content": r"Only \["},
    ]

    assert [(row["unit_id"], row["line"]) for row in rows(export_unit_markdown_escape_csv(units))] == [("a", "1"), ("b", "1"), ("b", "2")]
