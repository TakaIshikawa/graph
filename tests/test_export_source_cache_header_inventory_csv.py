from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_cache_header_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_cache_header_inventory_csv_reports_presence_and_dates():
    text = export_source_cache_header_inventory_csv(
        [
            {"id": "s2", "metadata": {"headers": {"ETag": "abc", "Cache-Control": "max-age=60"}}},
            {"id": "s1", "metadata": {"Last-Modified": "2024-01-01T00:00:00Z", "expires": "bad-date"}},
        ]
    )

    assert rows(text) == [
        {
            "source_id": "s1",
            "etag_present": "false",
            "last_modified": "2024-01-01T00:00:00+00:00",
            "cache_control_present": "false",
            "expires": "bad-date",
            "cache_header_bucket": "last_modified+expires",
        },
        {
            "source_id": "s2",
            "etag_present": "true",
            "last_modified": "",
            "cache_control_present": "true",
            "expires": "",
            "cache_header_bucket": "etag+cache_control",
        },
    ]


def test_export_source_cache_header_inventory_csv_path_mode(tmp_path):
    path = tmp_path / "cache.csv"
    stats = export_source_cache_header_inventory_csv([{"id": "s1"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["cache_header_bucket"] == "none"
    assert stats["source_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
