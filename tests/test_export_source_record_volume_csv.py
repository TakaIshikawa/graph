from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_source_record_volume_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_record_volume_csv_prefers_created_updated_then_metadata():
    text = export_source_record_volume_csv(
        [
            {"id": "b", "source_project": "p", "source_entity_type": "note", "created_at": "2024-01-03", "updated_at": "2024-02-01", "metadata": {}},
            {"id": "a", "source_project": "p", "source_entity_type": "note", "created_at": None, "updated_at": "2024-01-01", "metadata": {}},
            {"id": "c", "source_project": "p", "source_entity_type": "note", "created_at": "", "updated_at": "", "metadata": {"published": "2024-02-10"}},
        ],
        date_metadata_keys=["published"],
    )

    assert rows(text) == [
        {"source_project": "p", "source_entity_type": "note", "period": "2024-01", "unit_count": "2", "first_unit_id": "a", "last_unit_id": "b"},
        {"source_project": "p", "source_entity_type": "note", "period": "2024-02", "unit_count": "1", "first_unit_id": "c", "last_unit_id": "c"},
    ]


def test_export_source_record_volume_csv_year_path_mode(tmp_path):
    units = [{"id": "u", "source_project": "p", "source_entity_type": "n", "created_at": "2024-05-01"}]
    path = tmp_path / "volume.csv"

    stats = export_source_record_volume_csv(units, path, granularity="year")

    assert rows(path.read_text(encoding="utf-8"))[0]["period"] == "2024"
    assert stats["granularity"] == "year"
    assert stats["rows_exported"] == 1


def test_export_source_record_volume_csv_validates_granularity():
    with pytest.raises(ValueError, match="granularity"):
        export_source_record_volume_csv([], granularity="day")
