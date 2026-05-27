from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_html_comment_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_markdown_html_comment_csv_exports_single_and_same_line_comments():
    text = export_unit_markdown_html_comment_csv(
        [
            {"id": "u1", "title": "One", "content": "Intro <!-- first --> text\n<!-- second -->\nDone"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "u1", "title": "One", "line": "1", "comment_text": "first", "context": "Intro <!-- first --> text"},
        {"unit_id": "u1", "title": "One", "line": "2", "comment_text": "second", "context": "<!-- second -->"},
    ]


def test_unit_markdown_html_comment_csv_trims_multiline_comment_text():
    text = export_unit_markdown_html_comment_csv([{"id": "u1", "title": "One", "content": "A\n<!--\n hidden note \n-->\nB"}])

    assert rows(text)[0]["comment_text"] == "hidden note"
    assert rows(text)[0]["line"] == "2"


def test_unit_markdown_html_comment_csv_path_mode_writes_same_content(tmp_path):
    units = [{"id": "u1", "title": "One", "content": "<!-- note -->"}]
    path = tmp_path / "comments.csv"

    expected = export_unit_markdown_html_comment_csv(units)
    stats = export_unit_markdown_html_comment_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
