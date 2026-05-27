from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_html_data_attribute_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_multiple_attribute_value_styles_on_one_tag():
    content = "<div data-a=\"one\" data-b='two' data-c=three data-empty data-blank=\"\"></div>"

    found = rows(export_unit_html_data_attribute_csv([{"id": "u", "content": content}]))

    assert [(row["attribute"], row["value"]) for row in found] == [("data-a", "one"), ("data-b", "two"), ("data-blank", ""), ("data-c", "three"), ("data-empty", "")]


def test_fenced_code_is_excluded_and_file_writes(tmp_path):
    path = tmp_path / "attrs.csv"
    content = "```\n<div data-skip=\"x\"></div>\n```\n<span data-ok=yes></span>"

    result = export_unit_html_data_attribute_csv([{"id": "u", "content": content}], path)

    assert [(row["line_number"], row["attribute"]) for row in rows(path.read_text())] == [("4", "data-ok")]
    assert result["rows_exported"] == 1
