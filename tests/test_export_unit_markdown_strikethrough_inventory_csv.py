from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

from graph.export import export_units_to_markdown_strikethrough_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


@dataclass
class Unit:
    id: str
    content: str


def test_strikethrough_inventory_counts_lengths_and_samples():
    result = rows(export_units_to_markdown_strikethrough_inventory_csv([{"id": "b", "content": "Use ~~old term~~ and ~~ ~~.\n`~~code~~`"}]))[0]

    assert result == {
        "unit_id": "b",
        "strikethrough_span_count": "2",
        "total_strikethrough_text_length": "8",
        "empty_strikethrough_span_count": "1",
        "sample_texts": "old term",
    }


def test_strikethrough_inventory_ignores_fenced_code_and_sorts():
    result = rows(export_units_to_markdown_strikethrough_inventory_csv([{"id": "z", "content": "```md\n~~ignored~~\n```\n~~kept~~"}, {"id": "a", "content": ""}]))

    assert [row["unit_id"] for row in result] == ["a", "z"]
    assert result[1]["strikethrough_span_count"] == "1"
    assert result[1]["sample_texts"] == "kept"


def test_strikethrough_inventory_supports_object_units_and_path_write(tmp_path):
    output = tmp_path / "strike.csv"

    result = export_units_to_markdown_strikethrough_inventory_csv([Unit("o", "One ~~gone~~.")], output)

    assert result["path"] == str(output)
    assert result["unit_count"] == 1
    assert result["rows_exported"] == 1
    assert rows(output.read_text(encoding="utf-8"))[0]["sample_texts"] == "gone"
