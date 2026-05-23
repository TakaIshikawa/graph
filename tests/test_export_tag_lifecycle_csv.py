from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_tag_lifecycle_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_tag_lifecycle_summarizes_dates_and_undated_units():
    text = export_tag_lifecycle_csv(
        [
            {"id": "a", "tags": ["x", "y"], "metadata": {"published_at": "2024-01-01"}},
            {"id": "b", "tags": ["x"], "metadata": {"published_at": "2024-01-11"}},
            {"id": "c", "tags": ["x"], "metadata": {"published_at": "bad"}},
        ]
    )

    assert rows(text) == [
        {"tag": "x", "unit_count": "3", "first_seen": "2024-01-01", "last_seen": "2024-01-11", "active_span_days": "10", "undated_unit_count": "1"},
        {"tag": "y", "unit_count": "1", "first_seen": "2024-01-01", "last_seen": "2024-01-01", "active_span_days": "0", "undated_unit_count": "0"},
    ]


def test_tag_lifecycle_path_mode(tmp_path):
    path = tmp_path / "tags.csv"
    stats = export_tag_lifecycle_csv([{"id": "a", "tags": ["x"], "metadata": {}}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["undated_unit_count"] == "1"
    assert stats["unit_count"] == 1
