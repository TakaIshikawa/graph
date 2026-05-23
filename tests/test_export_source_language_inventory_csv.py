from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_language_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_language_inventory_csv_empty_input_returns_header():
    assert export_source_language_inventory_csv([]) == "language,collection,status,count,source_ids,source_keys\n"


def test_export_source_language_inventory_csv_groups_language_collection_and_status():
    text = export_source_language_inventory_csv(
        [
            {"id": "s1", "language": "en-US", "collection": "Inbox", "status": "active"},
            {"id": "s2", "metadata": {"lang": "EN_gb", "collection": "Inbox", "source_status": "active"}},
            {"id": "s3", "metadata": {"source_language": "ja-JP", "collection": "Archive", "status": "done"}},
        ]
    )

    assert rows(text) == [
        {
            "language": "en",
            "collection": "Inbox",
            "status": "active",
            "count": "2",
            "source_ids": "s1; s2",
            "source_keys": "lang; language",
        },
        {
            "language": "ja",
            "collection": "Archive",
            "status": "done",
            "count": "1",
            "source_ids": "s3",
            "source_keys": "source_language",
        },
    ]


def test_export_source_language_inventory_csv_uses_unknown_for_missing_language_dimensions():
    text = export_source_language_inventory_csv([{"source_id": "s1", "metadata": {"language": " "}}])

    assert rows(text) == [
        {
            "language": "unknown",
            "collection": "unknown",
            "status": "unknown",
            "count": "1",
            "source_ids": "s1",
            "source_keys": "missing",
        }
    ]


def test_export_source_language_inventory_csv_path_mode(tmp_path):
    path = tmp_path / "source-languages.csv"
    stats = export_source_language_inventory_csv([{"id": "s1", "language": "fr"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["language"] == "fr"
    assert stats["source_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
