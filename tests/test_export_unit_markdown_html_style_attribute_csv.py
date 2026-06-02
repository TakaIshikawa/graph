from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_markdown_html_style_attribute_csv import export_unit_markdown_html_style_attribute_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_style_attributes_and_ignores_comments_and_fences(tmp_path):
    units = [
        {
            "id": "u",
            "title": "Unit",
            "content": "<span style=\"color: red; font-weight: bold;\">A</span>\n<div style='display:block'>B</div>\n<p style=color:blue>C</p>\n<!-- <em style=\"bad: yes\"> -->\n```\n<strong style=\"skip: yes\">x</strong>\n```",
        }
    ]

    rows = _rows(export_unit_markdown_html_style_attribute_csv(units))

    assert rows == [
        {
            "unit_id": "u",
            "title": "Unit",
            "line_number": "1",
            "tag_name": "span",
            "style_text": "color: red; font-weight: bold;",
            "property_count": "2",
            "raw_tag": '<span style="color: red; font-weight: bold;">',
        },
        {"unit_id": "u", "title": "Unit", "line_number": "2", "tag_name": "div", "style_text": "display:block", "property_count": "1", "raw_tag": "<div style='display:block'>"},
        {"unit_id": "u", "title": "Unit", "line_number": "3", "tag_name": "p", "style_text": "color:blue", "property_count": "1", "raw_tag": "<p style=color:blue>"},
    ]
    output = tmp_path / "styles.csv"
    result = export_unit_markdown_html_style_attribute_csv(units, output)
    assert result["rows_exported"] == 3
    assert output.stat().st_size == result["bytes_written"]
