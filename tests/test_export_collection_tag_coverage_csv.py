from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_collection_tag_coverage_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_collection_tag_coverage_csv_empty_input_returns_header():
    assert export_collection_tag_coverage_csv([]) == (
        "collection,total_units,tagged_units,untagged_units,coverage_ratio,top_tags,sample_unit_ids\n"
    )


def test_export_collection_tag_coverage_csv_reports_coverage_and_top_tags():
    text = export_collection_tag_coverage_csv(
        [
            {"id": "u1", "metadata": {"collection": "Inbox"}, "tags": ["python", "ai"]},
            {"id": "u2", "metadata": {"collection": "Inbox"}, "tags": ["python"]},
            {"id": "u3", "metadata": {"collection": "Inbox"}, "tags": []},
            {"id": "u4", "metadata": {"collection": "Archive", "tags": ["ai"]}},
        ]
    )

    assert rows(text) == [
        {
            "collection": "Archive",
            "total_units": "1",
            "tagged_units": "1",
            "untagged_units": "0",
            "coverage_ratio": "1.00",
            "top_tags": "ai:1",
            "sample_unit_ids": "u4",
        },
        {
            "collection": "Inbox",
            "total_units": "3",
            "tagged_units": "2",
            "untagged_units": "1",
            "coverage_ratio": "0.67",
            "top_tags": "python:2; ai:1",
            "sample_unit_ids": "u1; u2; u3",
        },
    ]


def test_export_collection_tag_coverage_csv_uses_unassigned_for_missing_collection(tmp_path):
    path = tmp_path / "coverage.csv"
    stats = export_collection_tag_coverage_csv([{"id": "u1", "tags": []}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["collection"] == "unassigned"
    assert stats["unit_count"] == 1
