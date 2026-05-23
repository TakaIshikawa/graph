from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_outbound_link_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_outbound_link_inventory_csv_empty_input_returns_header():
    assert export_unit_outbound_link_inventory_csv([]) == "scheme,hostname,unit_count,link_count,unit_ids,sample_urls\n"


def test_export_unit_outbound_link_inventory_csv_groups_http_links_by_scheme_and_hostname():
    text = export_unit_outbound_link_inventory_csv(
        [
            {"id": "u1", "content": "See https://Example.test/a and https://example.test/b."},
            {"id": "u2", "content": "Visit http://example.test/c and not-a-url", "metadata": {"link": "https://other.test/x"}},
        ]
    )

    assert rows(text) == [
        {
            "scheme": "http",
            "hostname": "example.test",
            "unit_count": "1",
            "link_count": "1",
            "unit_ids": "u2",
            "sample_urls": "http://example.test/c",
        },
        {
            "scheme": "https",
            "hostname": "example.test",
            "unit_count": "1",
            "link_count": "2",
            "unit_ids": "u1",
            "sample_urls": "https://Example.test/a; https://example.test/b",
        },
        {
            "scheme": "https",
            "hostname": "other.test",
            "unit_count": "1",
            "link_count": "1",
            "unit_ids": "u2",
            "sample_urls": "https://other.test/x",
        },
    ]


def test_export_unit_outbound_link_inventory_csv_path_mode(tmp_path):
    path = tmp_path / "links.csv"
    stats = export_unit_outbound_link_inventory_csv([{"id": "u1", "content": "https://example.test"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["hostname"] == "example.test"
    assert stats["unit_count"] == 1
