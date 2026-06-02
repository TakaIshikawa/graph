from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_link_domain_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_link_domain_inventory_normalizes_www_and_keeps_linkless_units():
    text = export_units_to_link_domain_inventory_csv(
        [
            {"id": "b", "title": "Beta", "content": "https://www.Example.test/a https://example.test/b", "metadata": {"link": "https://other.test/x"}},
            {"id": "a", "title": "Alpha", "content": ""},
        ]
    )

    assert rows(text) == [
        {"unit_id": "a", "title": "Alpha", "domain_count": "0", "link_count": "0", "top_domain": "", "external_domain_count": "0"},
        {"unit_id": "b", "title": "Beta", "domain_count": "2", "link_count": "3", "top_domain": "example.test", "external_domain_count": "2"},
    ]


def test_link_domain_inventory_path_mode(tmp_path):
    path = tmp_path / "links.csv"
    stats = export_units_to_link_domain_inventory_csv([{"id": "a", "content": "https://example.test"}], path)

    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
