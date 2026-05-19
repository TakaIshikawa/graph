from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_transfer_candidates_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project="Bank", source_id=f"source-{unit_id}", source_entity_type="transaction", title=unit_id, content="", metadata=metadata or {}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_transfer_candidates_match_opposite_signed_amounts_within_window():
    text = export_unit_transfer_candidates_csv(
        [
            unit("out", metadata={"amount": "($100.00)", "currency": "USD", "account": "Checking", "date": "2026-05-01"}),
            unit("in", metadata={"amount": "$100.00", "currency": "USD", "account": "Savings", "date": "2026-05-01"}),
            unit("late", metadata={"amount": "$100.00", "currency": "USD", "account": "Brokerage", "date": "2026-05-08"}),
        ]
    )

    assert rows(text) == [
        {
            "source_unit_id": "out",
            "target_unit_id": "in",
            "amount": "100",
            "currency": "USD",
            "source_account": "Checking",
            "target_account": "Savings",
            "date_lag_days": "0",
            "confidence": "1",
            "evidence": "opposite signs, same amount and currency within 0 days",
        }
    ]


def test_transfer_candidates_skip_same_account_and_write_stats(tmp_path):
    units = [
        {"id": "a", "metadata": {"amount": "-50", "currency": "USD", "account": "Checking", "date": "2026-05-01"}},
        {"id": "b", "metadata": {"amount": "50", "currency": "USD", "account": "Checking", "date": "2026-05-02"}},
    ]
    expected = export_unit_transfer_candidates_csv(units)
    path = tmp_path / "transfers.csv"

    stats = export_unit_transfer_candidates_csv(units, path)

    assert rows(expected) == []
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 2
    assert stats["rows_exported"] == 0
    assert stats["bytes_written"] == path.stat().st_size
