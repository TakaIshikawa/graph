from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_date_conflicts_csv import export_unit_date_conflicts_csv
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, **overrides: object) -> KnowledgeUnit:
    data = {
        "id": unit_id,
        "source_project": "A",
        "source_id": unit_id,
        "source_entity_type": "note",
        "title": f"Title {unit_id}",
        "content": "content",
        "content_type": ContentType.INSIGHT,
        "metadata": {},
        "tags": [],
        "created_at": None,
        "updated_at": None,
        "ingested_at": None,
    }
    data.update(overrides)
    return KnowledgeUnit.model_construct(**data)


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_date_conflicts_csv_empty_input_has_header_only():
    assert export_unit_date_conflicts_csv([]) == (
        "unit_id,source_project,source_entity_type,earliest_date,latest_date,span_days,date_count,date_fields\n"
    )


def test_unit_date_conflicts_csv_flags_units_with_distinct_dates_over_minimum_span():
    text = export_unit_date_conflicts_csv(
        [
            unit(
                "a",
                created_at="2024-01-01T08:00:00Z",
                updated_at="2024-01-10",
                metadata={"published_at": "2024-01-03"},
            )
        ]
    )

    assert rows(text)[0] == {
        "unit_id": "a",
        "source_project": "A",
        "source_entity_type": "note",
        "earliest_date": "2024-01-01",
        "latest_date": "2024-01-10",
        "span_days": "9",
        "date_count": "3",
        "date_fields": "created_at=2024-01-01; metadata.published_at=2024-01-03; updated_at=2024-01-10",
    }


def test_unit_date_conflicts_csv_requires_two_distinct_dates_and_minimum_span():
    text = export_unit_date_conflicts_csv(
        [
            unit("a", created_at="2024-01-01", updated_at="2024-01-01"),
            unit("b", created_at="2024-01-01", updated_at="2024-01-02"),
        ],
        minimum_span_days=2,
    )

    assert rows(text) == []


def test_unit_date_conflicts_csv_ignores_malformed_metadata_dates():
    text = export_unit_date_conflicts_csv(
        [unit("a", created_at="2024-01-01", metadata={"date": "not-a-date", "observed_at": "2024-01-05"})]
    )

    assert rows(text)[0]["date_fields"] == "created_at=2024-01-01; metadata.observed_at=2024-01-05"


def test_unit_date_conflicts_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "unit-date-conflicts.csv"
    units = [unit("a", created_at="2024-01-01", updated_at="2024-01-03")]

    expected = export_unit_date_conflicts_csv(units)
    stats = export_unit_date_conflicts_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "conflict_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
