from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_collection_update_cadence_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_collection_update_cadence_csv_computes_stable_metrics():
    text = export_collection_update_cadence_csv(
        [
            {"id": "u1", "metadata": {"collection": "Inbox", "date": "2024-01-01T00:00:00Z"}},
            {"id": "u2", "metadata": {"collection": "Inbox", "date": "2024-01-04T00:00:00Z"}},
            {"id": "u3", "metadata": {"collection": "Inbox", "date": "2024-01-10T00:00:00Z"}},
        ],
        stale_after_days=30,
        reference_date="2024-02-15T00:00:00Z",
    )

    assert rows(text) == [
        {
            "collection": "Inbox",
            "update_count": "3",
            "first_seen": "2024-01-01T00:00:00+00:00",
            "last_seen": "2024-01-10T00:00:00+00:00",
            "median_gap_days": "4.50",
            "stale_after_days": "30",
            "is_stale": "true",
        }
    ]


def test_export_collection_update_cadence_csv_handles_single_undated_collection(tmp_path):
    path = tmp_path / "cadence.csv"
    stats = export_collection_update_cadence_csv([{"id": "u1"}], path, stale_after_days=7, reference_date="2024-01-01T00:00:00Z")

    assert rows(path.read_text(encoding="utf-8")) == [
        {
            "collection": "unassigned",
            "update_count": "0",
            "first_seen": "",
            "last_seen": "",
            "median_gap_days": "",
            "stale_after_days": "7",
            "is_stale": "unknown",
        }
    ]
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
