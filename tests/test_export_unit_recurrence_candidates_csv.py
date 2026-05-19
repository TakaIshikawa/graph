from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_recurrence_candidates_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, title: str, *, metadata: dict | None = None, source_project: object = "Bank") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="transaction",
        title=title,
        content="content",
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_recurrence_candidates_groups_normalized_titles_and_amounts():
    text = export_unit_recurrence_candidates_csv(
        [
            unit("b", "Gym, Inc.", metadata={"amount": "$10.00", "date": "2026-02-01"}),
            unit("a", "GYM INC", metadata={"amount": "10", "date": "2026-01-01"}),
            unit("c", "One Off", metadata={"date": "2026-01-01"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Bank",
            "recurrence_key": "gym inc",
            "unit_count": "2",
            "first_seen": "2026-01-01",
            "last_seen": "2026-02-01",
            "span_days": "31",
            "average_interval_days": "31.00",
            "interval_bucket": "monthly",
            "amount": "10",
            "representative_unit_ids": "a; b",
        }
    ]


def test_unit_recurrence_candidates_can_use_merchant_metadata_and_mapping_inputs():
    text = export_unit_recurrence_candidates_csv(
        [
            {"id": "a", "source_project": "Card", "title": "x", "metadata": {"merchant": "Coffee Shop", "date": "2026-05-01"}},
            {"id": "b", "source_project": "Card", "title": "y", "metadata": {"merchant": "Coffee-Shop", "date": "2026-05-08"}},
        ]
    )

    row = rows(text)[0]
    assert row["recurrence_key"] == "coffee shop"
    assert row["interval_bucket"] == "weekly"


def test_unit_recurrence_candidates_path_writes(tmp_path):
    units = [
        unit("a", "Rent", metadata={"date": "2026-01-01"}),
        unit("b", "Rent", metadata={"date": "2026-02-01"}),
    ]
    expected = export_unit_recurrence_candidates_csv(units)
    path = tmp_path / "recurrence.csv"
    stats = export_unit_recurrence_candidates_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 2
    assert stats["rows_exported"] == 1
