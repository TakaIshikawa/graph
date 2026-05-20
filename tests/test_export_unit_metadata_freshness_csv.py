from __future__ import annotations

import csv
from datetime import date, datetime
from io import StringIO

import pytest

from graph.export import export_unit_metadata_freshness_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_metadata_freshness_csv_parses_dates_and_lists():
    text = export_unit_metadata_freshness_csv(
        [
            {"id": "u1", "source_project": "p", "source_entity_type": "note", "metadata": {"seen": "2024-01-02T10:00:00Z", "bad": "nope"}},
            {"id": "u2", "source_project": "p", "source_entity_type": "note", "metadata": {"seen": [date(2024, 1, 5), "", datetime(2024, 1, 7, 4, 0)]}},
        ],
        reference_date=date(2024, 1, 10),
    )

    assert rows(text) == [
        {"source_project": "p", "source_entity_type": "note", "metadata_key": "seen", "observed_count": "3", "earliest_value": "2024-01-02", "latest_value": "2024-01-07", "days_since_latest": "3"}
    ]


def test_export_unit_metadata_freshness_csv_filters_by_min_count_and_path_mode(tmp_path):
    units = [{"id": "u1", "source_project": "p", "source_entity_type": "n", "metadata": {"one": "2024-01-01"}}]
    path = tmp_path / "fresh.csv"

    stats = export_unit_metadata_freshness_csv(units, path, reference_date="2024-01-02", min_count=2)

    assert path.read_text(encoding="utf-8") == "source_project,source_entity_type,metadata_key,observed_count,earliest_value,latest_value,days_since_latest\n"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 0
    assert stats["min_count"] == 2


def test_export_unit_metadata_freshness_csv_validates_limits():
    with pytest.raises(ValueError, match="min_count"):
        export_unit_metadata_freshness_csv([], min_count=0)
