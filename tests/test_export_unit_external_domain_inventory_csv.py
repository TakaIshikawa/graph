from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_external_domain_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_external_domain_inventory_csv_empty_input_has_header_only():
    assert export_unit_external_domain_inventory_csv([]) == "unit_id,title,domain,url_count,schemes,sample_urls\n"


def test_unit_external_domain_inventory_counts_duplicate_domains_and_mixed_schemes():
    text = export_unit_external_domain_inventory_csv(
        [
            {
                "id": "a",
                "title": "Alpha",
                "content": "See https://Example.test/a and http://example.test/b.",
                "metadata": {"links": ["ftp://files.example.test/drop", {"canonical": "https://example.test/c"}]},
            }
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "domain": "example.test",
            "url_count": "3",
            "schemes": "http; https",
            "sample_urls": "https://Example.test/a; http://example.test/b; https://example.test/c",
        },
        {
            "unit_id": "a",
            "title": "Alpha",
            "domain": "files.example.test",
            "url_count": "1",
            "schemes": "ftp",
            "sample_urls": "ftp://files.example.test/drop",
        },
    ]


def test_unit_external_domain_inventory_handles_missing_content_and_metadata(tmp_path):
    path = tmp_path / "domains.csv"
    data = [{"id": "a", "title": "No links", "content": None, "metadata": None}]

    stats = export_unit_external_domain_inventory_csv(data, path)

    assert rows(path.read_text(encoding="utf-8")) == []
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 0
