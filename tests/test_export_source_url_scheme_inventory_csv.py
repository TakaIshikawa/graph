from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_url_scheme_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_url_scheme_inventory_normalizes_schemes_and_missing_scheme():
    text = export_source_url_scheme_inventory_csv(
        [
            {
                "id": "s1",
                "name": "Source",
                "url": "HTTPS://example.test/a",
                "metadata": {"links": ["mailto:a@example.test", "example.org/path"]},
            }
        ]
    )

    assert rows(text) == [
        {"source_id": "s1", "source_name": "Source", "scheme": "https", "url_count": "1", "sample_urls": "HTTPS://example.test/a"},
        {"source_id": "s1", "source_name": "Source", "scheme": "mailto", "url_count": "1", "sample_urls": "mailto:a@example.test"},
        {"source_id": "s1", "source_name": "Source", "scheme": "missing", "url_count": "1", "sample_urls": "example.org/path"},
    ]


def test_source_url_scheme_inventory_path_mode(tmp_path):
    path = tmp_path / "source-schemes.csv"
    stats = export_source_url_scheme_inventory_csv([{"source_id": "s1", "title": "S", "url": "https://example.test"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["scheme"] == "https"
    assert stats["source_count"] == 1
