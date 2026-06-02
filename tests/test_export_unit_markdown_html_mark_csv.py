from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_mark_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_mark_elements_with_attributes():
    result = rows(export_units_to_markdown_html_mark_csv([{"id": "u", "title": "T", "source": "s", "content": '<mark class="hot" data-x="1">Important</mark>'}]))

    assert result[0] == {"unit_id": "u", "title": "T", "source": "s", "line_number": "1", "marked_text": "Important", "raw_attributes": 'class="hot" data-x="1"', "class_attribute": "hot", "closed": "True"}


def test_reports_unclosed_and_ignores_fenced_code():
    content = "```html\n<mark>skip</mark>\n```\n<mark>open"

    result = rows(export_units_to_markdown_html_mark_csv([{"id": "u", "content": content}]))

    assert [(row["marked_text"], row["closed"]) for row in result] == [("open", "False")]


def test_path_write_returns_export_metadata(tmp_path):
    output = tmp_path / "mark.csv"

    result = export_units_to_markdown_html_mark_csv([{"id": "u", "content": "<mark>x</mark>"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
