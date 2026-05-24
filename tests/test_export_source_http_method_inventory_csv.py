from __future__ import annotations

import csv
from io import StringIO
from types import SimpleNamespace

from graph.export import export_source_http_method_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_http_method_inventory_normalizes_and_groups_methods():
    text = export_source_http_method_inventory_csv(
        [
            {"id": "s1", "url": "https://a.test", "metadata": {"method": "get"}},
            SimpleNamespace(id="s2", url="https://b.test", metadata={"request": {"http_method": "post"}}),
            {"id": "s3"},
        ]
    )

    by_method = {row["http_method"]: row for row in rows(text)}
    assert by_method["GET"]["source_ids"] == "s1"
    assert by_method["POST"]["urls"] == "https://b.test"
    assert by_method["UNKNOWN"]["count"] == "1"


def test_source_http_method_inventory_writes_path_metadata(tmp_path):
    path = tmp_path / "methods.csv"
    stats = export_source_http_method_inventory_csv([{"id": "s1", "method": "head"}], path)

    assert stats["source_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
