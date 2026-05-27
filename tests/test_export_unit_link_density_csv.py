from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_link_density_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_link_density_counts_link_types_and_density():
    text = export_units_to_link_density_csv(
        [{"id": "a", "title": "Alpha", "content": "[[Home]] [site](https://example.test/a) https://raw.test/b"}]
    )

    row = rows(text)[0]
    assert row["internal_wikilink_count"] == "1"
    assert row["markdown_link_count"] == "1"
    assert row["raw_url_count"] == "1"
    assert row["total_link_count"] == "3"
    assert row["links_per_1000_chars"] == "51.72"


def test_unit_link_density_handles_empty_content():
    assert rows(export_units_to_link_density_csv([{"id": "a", "content": None}]))[0] == {
        "unit_id": "a",
        "title": "",
        "content_length": "0",
        "internal_wikilink_count": "0",
        "markdown_link_count": "0",
        "raw_url_count": "0",
        "total_link_count": "0",
        "links_per_1000_chars": "0.00",
    }
