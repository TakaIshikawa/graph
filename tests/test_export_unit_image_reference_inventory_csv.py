from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_image_reference_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_image_reference_inventory_counts_markdown_html_metadata_and_location(tmp_path):
    unit = {"id": "u", "content": "![](local.png) ![Alt](https://e/x.png)\n<img src='http://e/y.jpg'>", "metadata": {"images": ["cover.jpg", {"icon": "nested.png"}], "cover": "https://e/c.png"}}
    output = tmp_path / "images.csv"
    result = export_units_to_image_reference_inventory_csv([unit], output)
    row = rows(output.read_text(encoding="utf-8"))[0]

    assert result["unit_count"] == 1
    assert row["markdown_image_count"] == "2"
    assert row["html_image_count"] == "1"
    assert row["metadata_image_count"] == "3"
    assert row["missing_alt_count"] == "1"
    assert row["remote_image_count"] == "3"
    assert row["local_image_count"] == "3"
