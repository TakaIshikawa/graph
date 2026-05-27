from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_markdown_admonition_inventory_csv import export_units_to_markdown_admonition_inventory_csv


def test_export_units_to_markdown_admonition_inventory_csv_detects_common_syntaxes():
    rows = list(csv.DictReader(StringIO(export_units_to_markdown_admonition_inventory_csv([{"id": "u1", "title": "Doc", "content": "> [!NOTE] Read\n!!! warning \"Careful\"\n:::tip Try"}]))))

    assert [(row["admonition_type"], row["syntax"], row["line_number"], row["preview"]) for row in rows] == [
        ("note", "obsidian", "1", "Read"),
        ("warning", "mkdocs", "2", "Careful"),
        ("tip", "container", "3", "Try"),
    ]
