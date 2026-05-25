from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_outbound_reference_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_outbound_reference_extracts_and_deduplicates_references():
    result = rows(export_units_to_outbound_reference_csv([
        {"id": "u", "content": "See [A](https://Example.com/a), https://example.com/a [[Topic]] @smith2020", "metadata": {"url": "https://other.test/x"}}
    ]))[0]

    assert result["domains"] == "example.com; other.test"
    assert result["wikilinks"] == "Topic"
    assert result["citations"] == "@smith2020"
    assert result["reference_count"] == "4"
