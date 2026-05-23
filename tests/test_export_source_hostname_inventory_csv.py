from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_hostname_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_hostname_inventory_csv_empty_input_returns_header():
    assert export_source_hostname_inventory_csv([]) == "hostname,source_count,source_ids,sample_urls\n"


def test_export_source_hostname_inventory_csv_groups_hostnames_and_sorts_by_count():
    text = export_source_hostname_inventory_csv(
        [
            {"id": "s1", "url": "https://Example.test/a"},
            {"id": "s2", "source_url": "http://example.test/b"},
            {"id": "s3", "metadata": {"canonical_url": "https://other.test/c"}},
            {"id": "s4", "url": "notaurl"},
            {"id": "s5", "url": ""},
        ]
    )

    assert rows(text) == [
        {
            "hostname": "example.test",
            "source_count": "2",
            "source_ids": "s1; s2",
            "sample_urls": "https://Example.test/a; http://example.test/b",
        },
        {
            "hostname": "unknown",
            "source_count": "2",
            "source_ids": "s4; s5",
            "sample_urls": "notaurl",
        },
        {
            "hostname": "other.test",
            "source_count": "1",
            "source_ids": "s3",
            "sample_urls": "https://other.test/c",
        },
    ]


def test_export_source_hostname_inventory_csv_path_mode(tmp_path):
    path = tmp_path / "hostnames.csv"
    stats = export_source_hostname_inventory_csv([{"id": "s1", "url": "example.test/path"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["hostname"] == "example.test"
    assert stats["source_count"] == 1
