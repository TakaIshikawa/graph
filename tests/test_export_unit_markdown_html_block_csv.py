from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_block_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_block_html_tags_with_attributes():
    content = '<div class="note" id="n1">Text</div>\n<section data-x="1">'

    result = rows(export_units_to_markdown_html_block_csv([{"id": "u", "title": "T", "source": "s", "content": content}]))

    assert [(row["tag_name"], row["class_attribute"], row["id_value"], row["self_contained"]) for row in result] == [
        ("div", "note", "n1", "True"),
        ("section", "", "", "False"),
    ]
    assert result[0]["source"] == "s"


def test_ignores_inline_html_and_fenced_code():
    content = "Text <div>inline</div>\n```html\n<div>skip</div>\n```\n<table><tr></tr></table>"

    result = rows(export_units_to_markdown_html_block_csv([{"id": "u", "content": content}]))

    assert [(row["tag_name"], row["line_number"]) for row in result] == [("table", "5")]


def test_path_write_returns_export_metadata(tmp_path):
    output = tmp_path / "html_block.csv"

    result = export_units_to_markdown_html_block_csv([{"id": "u", "content": "<aside>x</aside>"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
