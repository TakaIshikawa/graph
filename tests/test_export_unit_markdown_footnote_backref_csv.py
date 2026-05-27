from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_footnote_backref_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_numeric_named_and_multiple_backrefs():
    content = '<a href="#fnref1">↩</a> <a href="#fnref-note">back</a> <a href="#other">x</a>'

    found = rows(export_unit_markdown_footnote_backref_csv([{"id": "u", "content": content}]))

    assert [(row["footnote_id"], row["href"], row["backref_text"]) for row in found] == [("1", "#fnref1", "↩"), ("-note", "#fnref-note", "back")]


def test_sorting_fenced_code_and_file_writing(tmp_path):
    path = tmp_path / "backrefs.csv"
    units = [{"id": "b", "content": "```\n<a href=\"#fnrefskip\">x</a>\n```\n<a href=\"#fnrefb\">b</a>"}, {"id": "a", "content": "<a href=\"#fnrefa\">a</a>"}]

    result = export_unit_markdown_footnote_backref_csv(units, path)
    found = rows(path.read_text())

    assert [row["unit_id"] for row in found] == ["a", "b"]
    assert [row["footnote_id"] for row in found] == ["a", "b"]
    assert result["rows_exported"] == 2
