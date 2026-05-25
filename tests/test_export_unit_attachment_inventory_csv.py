from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_attachment_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_attachment_inventory_empty_input_has_header():
    assert export_units_to_attachment_inventory_csv([]).startswith("unit_id,attachment_count,attachment_paths")


def test_attachment_inventory_normalizes_and_deduplicates_paths():
    result = rows(export_units_to_attachment_inventory_csv([
        {"id": "b", "metadata": {"attachments": ["docs/a.pdf", {"path": "images/a.png"}, {"url": "docs/a.pdf"}]}},
        {"id": "a", "metadata": {}},
    ]))

    assert [row["unit_id"] for row in result] == ["a", "b"]
    assert result[0]["missing_attachment_metadata"] == "true"
    assert result[1]["attachment_count"] == "2"
    assert result[1]["attachment_paths"] == "docs/a.pdf; images/a.png"
    assert result[1]["attachment_extensions"] == ".pdf; .png"
    assert result[1]["has_images"] == "true"
    assert result[1]["has_documents"] == "true"


def test_attachment_inventory_path_mode(tmp_path):
    path = tmp_path / "attachments.csv"
    stats = export_units_to_attachment_inventory_csv([{"id": "u", "metadata": {"files": "one.txt"}}], path)
    assert rows(path.read_text(encoding="utf-8"))[0]["attachment_count"] == "1"
    assert stats["rows_exported"] == 1
