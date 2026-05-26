from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_collection_member_recency_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_collection_member_recency_tracks_valid_invalid_multiple_and_stale():
    result = rows(
        export_collection_member_recency_csv(
            [
                {"id": "a", "updated_at": "2024-01-01T00:00:00Z", "collections": ["A", "B"]},
                {"id": "b", "metadata": {"updated_at": "bad", "collection": "A"}},
                {"id": "c", "updated_at": "2024-03-01T00:00:00Z", "collection": "A"},
            ],
            stale_before="2024-02-01T00:00:00Z",
        )
    )

    parsed = {row["collection"]: row for row in result}
    assert parsed["A"]["unit_count"] == "3"
    assert parsed["A"]["stale_unit_count"] == "1"
    assert parsed["A"]["missing_timestamp_count"] == "1"
    assert parsed["B"]["latest_updated_at"] == "2024-01-01T00:00:00+00:00"


def test_collection_member_recency_writes_metadata(tmp_path):
    output = tmp_path / "recency.csv"
    result = export_collection_member_recency_csv([{"id": "u"}], output)

    assert result["unit_count"] == 1
    assert result["bytes_written"] == output.stat().st_size
