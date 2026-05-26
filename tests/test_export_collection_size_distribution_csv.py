from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_collection_size_distribution_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_collection_size_distribution_counts_multi_collections_and_unassigned():
    result = rows(
        export_collection_size_distribution_csv(
            [
                {"id": "a", "content": "1234", "metadata": {"collections": ["Work", "Ideas"], "source": "s1"}},
                {"id": "b", "content": "12", "collection": "Work", "source_project": "s2"},
                {"id": "c", "content": ""},
            ]
        )
    )

    parsed = {row["collection"]: row for row in result}
    assert parsed["Work"]["unit_count"] == "2"
    assert parsed["Work"]["source_count"] == "2"
    assert parsed["Work"]["average_content_length"] == "3.00"
    assert parsed["unassigned"]["unit_count"] == "1"


def test_collection_size_distribution_writes_metadata(tmp_path):
    output = tmp_path / "collections.csv"
    result = export_collection_size_distribution_csv([{"id": "u"}], output)

    assert result["rows_exported"] == 1
    assert result["bytes_written"] == output.stat().st_size
