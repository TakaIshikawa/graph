from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_metadata_timestamp_coverage_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_metadata_timestamp_coverage_reports_parseability_and_precision():
    text = export_unit_metadata_timestamp_coverage_csv(
        [
            {
                "id": "a",
                "title": "Alpha",
                "metadata": {"published_at": "2024-01-02T03:04:05Z", "updated_date": "2024-02-03", "created_at": "bad"},
            }
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "timestamp_keys_present": "created_at; published_at; updated_date",
            "parsed_timestamp_count": "2",
            "earliest_timestamp": "2024-01-02T03:04:05+00:00",
            "latest_timestamp": "2024-02-03T00:00:00",
            "precision_summary": "date:1; datetime:1; invalid:1",
        }
    ]


def test_unit_metadata_timestamp_coverage_handles_missing_metadata_and_path_mode(tmp_path):
    path = tmp_path / "timestamps.csv"

    stats = export_unit_metadata_timestamp_coverage_csv([{"id": "a", "title": "Alpha", "metadata": None}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["parsed_timestamp_count"] == "0"
    assert stats["unit_count"] == 1
