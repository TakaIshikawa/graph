from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_source_field_nullability_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_field_nullability_csv_core_and_metadata_fields():
    text = export_source_field_nullability_csv(
        [
            {"id": "u1", "source_project": "p", "source_entity_type": "note", "title": "", "content": "body", "source_id": "s1", "created_at": None, "updated_at": "2024", "metadata": {"url": ""}},
            {"id": "u2", "source_project": "p", "source_entity_type": "note", "title": "T", "content": [], "source_id": "s2", "created_at": "2024", "updated_at": None, "metadata": {"url": "x"}},
        ],
        metadata_keys=["url"],
        min_blank_percent=50.0,
    )

    data = {row["field_name"]: row for row in rows(text)}
    assert data["content"]["blank_percent"] == "50.00"
    assert data["metadata:url"]["blank_count"] == "1"
    assert "source_id" not in data


def test_export_source_field_nullability_csv_path_mode(tmp_path):
    units = [{"source_project": "p", "source_entity_type": "n", "title": None, "content": "", "source_id": "", "created_at": None, "updated_at": None, "metadata": {}}]
    path = tmp_path / "nulls.csv"

    stats = export_source_field_nullability_csv(units, path)

    assert len(rows(path.read_text(encoding="utf-8"))) == 5
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 5


def test_export_source_field_nullability_csv_validates_percent():
    with pytest.raises(ValueError, match="min_blank_percent"):
        export_source_field_nullability_csv([], min_blank_percent=101)
