from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_outbound_domain_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_outbound_domain_inventory_counts_http_urls_from_content_and_metadata():
    text = export_units_to_outbound_domain_inventory_csv(
        [
            {"id": "a", "title": "Alpha", "content": "https://Example.test/a https://example.test/b", "metadata": {"url": "http://other.test/c"}},
        ]
    )

    assert rows(text) == [
        {"unit_id": "a", "title": "Alpha", "domain": "example.test", "url_count": "2", "sample_url": "https://Example.test/a"},
        {"unit_id": "a", "title": "Alpha", "domain": "other.test", "url_count": "1", "sample_url": "http://other.test/c"},
    ]


def test_outbound_domain_inventory_path_mode(tmp_path):
    path = tmp_path / "domains.csv"
    stats = export_units_to_outbound_domain_inventory_csv([{"id": "a", "content": "https://example.test"}], path)

    assert stats["rows_exported"] == 1
    assert path.read_text(encoding="utf-8").startswith("unit_id,title,domain,url_count,sample_url")
