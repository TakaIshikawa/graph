from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_external_link_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_external_link_inventory_empty_input_returns_header():
    assert export_units_to_external_link_inventory_csv([]) == "unit_id,title,url,domain,source_field,occurrence_count\n"


def test_unit_external_link_inventory_counts_content_duplicates_and_sorts():
    text = export_units_to_external_link_inventory_csv(
        [
            {"id": "b", "title": "Beta", "content": "See [[Internal]] and https://b.test/x mailto:a@b.test."},
            {"id": "a", "title": "Alpha", "content": "https://Example.test/a https://Example.test/a http://z.test/end."},
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "url": "http://z.test/end",
            "domain": "z.test",
            "source_field": "content",
            "occurrence_count": "1",
        },
        {
            "unit_id": "a",
            "title": "Alpha",
            "url": "https://Example.test/a",
            "domain": "example.test",
            "source_field": "content",
            "occurrence_count": "2",
        },
        {
            "unit_id": "b",
            "title": "Beta",
            "url": "https://b.test/x",
            "domain": "b.test",
            "source_field": "content",
            "occurrence_count": "1",
        },
    ]


def test_unit_external_link_inventory_includes_metadata_fields_and_path_mode(tmp_path):
    path = tmp_path / "external.csv"
    stats = export_units_to_external_link_inventory_csv(
        [{"id": "a", "title": "Alpha", "metadata": {"url": "https://example.test", "source": {"canonical_url": "ftp://skip.test"}}}],
        path,
    )

    assert rows(path.read_text(encoding="utf-8")) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "url": "https://example.test",
            "domain": "example.test",
            "source_field": "metadata.url",
            "occurrence_count": "1",
        }
    ]
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
