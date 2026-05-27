from __future__ import annotations

import csv
from io import StringIO
from types import SimpleNamespace

from graph.export import export_units_to_callout_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_callout_inventory_counts_types_titles_folding_and_lines():
    text = "> [!NOTE]\n> body\n> more\n\n> [! warning]- Important\n> body\n> ordinary quote"
    result = rows(export_units_to_callout_inventory_csv([{"id": "b", "content": ""}, {"id": "a", "content": text}]))

    assert [row["unit_id"] for row in result] == ["a", "b"]
    assert result[0]["callout_count"] == "2"
    assert result[0]["callout_types"] == "note; warning"
    assert result[0]["folded_callout_count"] == "1"
    assert result[0]["titled_callout_count"] == "1"
    assert result[0]["max_callout_line_count"] == "3"
    assert result[1]["callout_count"] == "0"


def test_callout_inventory_supports_objects_and_path_write(tmp_path):
    output = tmp_path / "callouts.csv"
    result = export_units_to_callout_inventory_csv([SimpleNamespace(id="u", content="> [!TIP]+ Title")], output)

    assert result["unit_count"] == 1
    assert result["rows_exported"] == 1
    assert result["bytes_written"] == output.stat().st_size
    assert rows(output.read_text(encoding="utf-8"))[0]["callout_types"] == "tip"
