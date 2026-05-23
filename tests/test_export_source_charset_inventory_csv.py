from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_charset_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_charset_inventory_csv_empty_input_returns_header():
    assert export_source_charset_inventory_csv([]) == "charset,source_count,source_ids,source_keys\n"


def test_export_source_charset_inventory_csv_groups_charset_hints():
    text = export_source_charset_inventory_csv(
        [
            {"id": "s2", "metadata": {"content_type": "text/html; charset=UTF_8"}},
            {"id": "s1", "charset": "utf-8"},
            {"id": "s3", "metadata": {"headers": {"Content-Type": "application/json; charset=Shift_JIS"}}},
        ]
    )

    assert rows(text) == [
        {"charset": "shift-jis", "source_count": "1", "source_ids": "s3", "source_keys": "headers.Content-Type"},
        {"charset": "utf-8", "source_count": "2", "source_ids": "s1; s2", "source_keys": "charset; content_type"},
    ]


def test_export_source_charset_inventory_csv_uses_unknown_for_missing_charset(tmp_path):
    path = tmp_path / "charsets.csv"
    stats = export_source_charset_inventory_csv([{"source_id": "s1", "metadata": {"title": "No charset"}}], path)

    assert rows(path.read_text(encoding="utf-8")) == [
        {"charset": "unknown", "source_count": "1", "source_ids": "s1", "source_keys": "missing"}
    ]
    assert stats["source_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
