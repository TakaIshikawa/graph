from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

from graph.export import export_units_to_markdown_custom_id_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


@dataclass
class Unit:
    id: str
    content: str


def test_custom_id_inventory_counts_ids_duplicates_and_attributes():
    result = rows(export_units_to_markdown_custom_id_inventory_csv([{"id": "u1", "content": "# Intro {#intro .lead key=value}\n{#intro .standalone}"}]))[0]

    assert result == {
        "unit_id": "u1",
        "custom_id_count": "2",
        "duplicate_custom_id_count": "1",
        "class_attribute_count": "2",
        "key_value_attribute_count": "1",
        "custom_ids": "intro",
    }


def test_custom_id_inventory_ignores_fenced_code():
    result = rows(export_units_to_markdown_custom_id_inventory_csv([{"id": "u1", "content": "```md\n# X {#ignored .x y=z}\n```\n## Kept {#kept}"}]))[0]

    assert result["custom_id_count"] == "1"
    assert result["custom_ids"] == "kept"


def test_custom_id_inventory_supports_object_units_and_path_write(tmp_path):
    output = tmp_path / "ids.csv"

    result = export_units_to_markdown_custom_id_inventory_csv([Unit("o", "# A {#a .c}")], output)

    assert result["path"] == str(output)
    assert rows(output.read_text(encoding="utf-8"))[0]["custom_ids"] == "a"
