from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_html_element_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_html_element_inventory_ignores_fences_and_counts_unsafe(tmp_path):
    text = "<A href='/x'>x</a><img src=x><script></script>\n```\n<form></form>\n```"
    output = tmp_path / "html.csv"
    result = export_units_to_html_element_inventory_csv([{"id": "u", "content": text}], output)
    row = rows(output.read_text(encoding="utf-8"))[0]

    assert result["bytes_written"] == output.stat().st_size
    assert row["html_element_count"] == "5"
    assert row["distinct_tags"] == "a; img; script"
    assert row["link_tag_count"] == "2"
    assert row["image_tag_count"] == "1"
    assert row["unsafe_tag_count"] == "2"
