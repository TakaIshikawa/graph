from __future__ import annotations

import csv
from io import StringIO
from types import SimpleNamespace

from graph.export import export_unit_attachment_size_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_attachment_size_inventory_parses_direct_and_nested_sizes():
    text = export_unit_attachment_size_inventory_csv(
        [
            {"id": "u1", "metadata": {"attachments": [{"size": "1 KB"}, {"bytes": 2048}], "file_size": "2 MB"}},
            SimpleNamespace(id="u2", metadata={}),
        ]
    )

    by_id = {row["unit_id"]: row for row in rows(text)}
    assert by_id["u1"]["attachment_count"] == "3"
    assert by_id["u1"]["total_bytes"] == "2003048"
    assert by_id["u1"]["largest_bytes"] == "2000000"
    assert by_id["u1"]["size_bucket"] == "medium"
    assert by_id["u2"]["size_bucket"] == "missing"


def test_unit_attachment_size_inventory_writes_path_metadata(tmp_path):
    path = tmp_path / "attachments.csv"
    stats = export_unit_attachment_size_inventory_csv([{"id": "u1", "metadata": {"bytes": 1}}], path)

    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
