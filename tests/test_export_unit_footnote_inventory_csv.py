from __future__ import annotations

import csv
from io import StringIO
from types import SimpleNamespace

from graph.export import export_units_to_footnote_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_footnote_inventory_counts_definitions_references_and_gaps():
    text = "[^A]: first\nBody [^a] [^missing]\n[^A]: duplicate\n[^unused]: nope"
    row = rows(export_units_to_footnote_inventory_csv([{"id": "u", "content": text}]))[0]

    assert row["footnote_definition_count"] == "3"
    assert row["footnote_reference_count"] == "2"
    assert row["unresolved_reference_count"] == "1"
    assert row["unused_definition_count"] == "1"
    assert row["duplicate_definition_count"] == "1"


def test_footnote_inventory_objects_and_path_write(tmp_path):
    output = tmp_path / "footnotes.csv"
    result = export_units_to_footnote_inventory_csv([SimpleNamespace(id="u", content="[^x]: def")], output)

    assert result["bytes_written"] == output.stat().st_size
    assert rows(output.read_text(encoding="utf-8"))[0]["footnote_reference_count"] == "0"
