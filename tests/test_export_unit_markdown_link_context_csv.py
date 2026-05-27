from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export.unit_markdown_link_context_csv import export_units_to_markdown_link_context_csv


def test_export_units_to_markdown_link_context_csv_extracts_inline_reference_and_definitions_not_images():
    rows = list(csv.DictReader(StringIO(export_units_to_markdown_link_context_csv([{"id": "u1", "content": "See [One](https://one.test) and ![Img](x.png).\nUse [Two][ref].\n[ref]: https://two.test"}], context_chars=4))))

    assert [(row["label"], row["target"], row["line_number"]) for row in rows] == [
        ("One", "https://one.test", "1"),
        ("Two", "https://two.test", "2"),
        ("ref", "https://two.test", "3"),
    ]
    assert rows[0]["context_before"] == "See"


def test_export_units_to_markdown_link_context_csv_validates_context_chars():
    with pytest.raises(ValueError):
        export_units_to_markdown_link_context_csv([], context_chars=-1)
