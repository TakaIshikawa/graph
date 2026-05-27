from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_timestamp_field_coverage_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_timestamp_field_coverage_counts_top_level_and_metadata():
    text = export_source_timestamp_field_coverage_csv(
        [
            {"source_project": "web", "created_at": "2024-01-01", "metadata": {"updated_at": "bad"}},
            {"source_project": "web", "metadata": {"created_at": "2024-02-01T00:00:00Z"}},
        ],
        fields=["created_at", "updated_at"],
    )

    assert rows(text) == [
        {"source_project": "web", "field": "created_at", "unit_count": "2", "present_count": "2", "coverage_ratio": "1.00", "invalid_count": "0"},
        {"source_project": "web", "field": "updated_at", "unit_count": "2", "present_count": "1", "coverage_ratio": "0.50", "invalid_count": "1"},
    ]


def test_source_timestamp_field_coverage_default_fields_and_path_mode(tmp_path):
    path = tmp_path / "timestamps.csv"
    stats = export_source_timestamp_field_coverage_csv([{"metadata": {"published_at": "2024-01-01"}}], path)

    exported = rows(path.read_text(encoding="utf-8"))
    assert [row["field"] for row in exported] == ["created_at", "updated_at", "published_at", "archived_at"]
    assert exported[2]["present_count"] == "1"
    assert stats["rows_exported"] == 4
