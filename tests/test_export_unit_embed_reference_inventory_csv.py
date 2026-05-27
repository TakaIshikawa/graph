from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_embed_reference_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_embed_reference_inventory_parses_alias_sections_files_and_targets(tmp_path):
    text = "![[Note]] [[Not embed]] ![[file.pdf]] ![[Note#Part|Alias]] ![[note]]"
    output = tmp_path / "embeds.csv"
    result = export_units_to_embed_reference_inventory_csv([{"id": "u", "content": text}], output)
    row = rows(output.read_text(encoding="utf-8"))[0]

    assert result["rows_exported"] == 1
    assert row["embed_count"] == "4"
    assert row["note_embed_count"] == "3"
    assert row["file_embed_count"] == "1"
    assert row["section_embed_count"] == "1"
    assert row["alias_embed_count"] == "1"
    assert row["distinct_targets"] == "file.pdf; Note"
