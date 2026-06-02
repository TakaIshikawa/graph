from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_underline_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_underline_elements_with_attributes():
    result = rows(export_units_to_markdown_html_underline_csv([{"id": "u", "title": "T", "source": "s", "content": '<u class="term" data-x="1">Term</u>'}]))

    assert result[0]["underlined_text"] == "Term"
    assert result[0]["raw_attributes"] == 'class="term" data-x="1"'
    assert result[0]["class_attribute"] == "term"
    assert result[0]["closed"] == "True"
    assert result[0]["source"] == "s"


def test_reports_unclosed_and_ignores_fenced_code():
    content = "```html\n<u>skip</u>\n```\n<u>open"

    result = rows(export_units_to_markdown_html_underline_csv([{"id": "u", "content": content}]))

    assert [(row["underlined_text"], row["closed"]) for row in result] == [("open", "False")]


def test_path_write_returns_export_metadata(tmp_path):
    output = tmp_path / "underline.csv"

    result = export_units_to_markdown_html_underline_csv([{"id": "u", "content": "<u>x</u>"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
