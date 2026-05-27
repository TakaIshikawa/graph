from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_setext_heading_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_level_one_and_two_headings_with_boundaries():
    content = "Title\n=====\n\nParagraph\n---\n\n---"

    found = rows(export_unit_markdown_setext_heading_csv([{"id": "u", "content": content}]))

    assert [(row["line_number"], row["level"], row["text"], row["underline"]) for row in found] == [("1", "1", "Title", "====="), ("4", "2", "Paragraph", "---")]


def test_fenced_code_excluded_sorting_and_write_metadata(tmp_path):
    path = tmp_path / "setext.csv"
    units = [{"id": "b", "content": "```\nCode\n---\n```\nReal\n---"}, {"id": "a", "content": "A\n==="}]

    result = export_unit_markdown_setext_heading_csv(units, path)
    found = rows(path.read_text())

    assert [row["unit_id"] for row in found] == ["a", "b"]
    assert found[1]["line_number"] == "5"
    assert result["rows_exported"] == 2
