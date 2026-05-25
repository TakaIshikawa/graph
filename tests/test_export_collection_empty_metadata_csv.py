from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_collection_empty_metadata_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_collection_empty_metadata_counts_missing_fields_and_unassigned_bucket():
    result = {row["collection"]: row for row in rows(export_collection_empty_metadata_csv([
        {"id": "u1", "title": "", "content": "", "metadata": {"collection": "c1"}},
        {"id": "u2", "title": "Ok", "content": "body", "tags": ["x"], "source_id": "s", "created_at": "2024-01-01"},
        {"id": "u3", "title": "Ok", "content": "body", "metadata": {"collections": ["c1", "c2"]}},
    ]))}

    assert result["unassigned"]["total_units"] == "1"
    assert result["c1"]["total_units"] == "2"
    assert result["c1"]["blank_content"] == "1"
    assert result["c1"]["missing_rate"] == "0.80"
