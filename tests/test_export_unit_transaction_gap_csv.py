from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export.unit_transaction_gap_csv import export_units_to_transaction_gap_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, source_project: str = "Bank") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="transaction",
        title=unit_id,
        content="",
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_transaction_gap_reports_only_gaps_greater_than_minimum_partitioned_by_account_and_currency():
    text = export_units_to_transaction_gap_csv(
        [
            unit("usd-1", metadata={"amount": "1", "account": "Checking", "currency": "USD", "date": "2026-01-01"}, source_project="Bank A"),
            unit("eur-1", metadata={"amount": "1", "account": "Checking", "currency": "EUR", "date": "2026-01-15"}),
            unit("other-account", metadata={"amount": "1", "account": "Savings", "currency": "USD", "date": "2026-02-01"}),
            unit("usd-2", metadata={"amount": "-1", "account": "Checking", "currency": "USD", "date": "2026-01-20"}),
            unit("usd-3", metadata={"amount": "1", "account": "Checking", "currency": "USD", "date": "2026-03-01"}, source_project="Bank B"),
            unit("usd-4", metadata={"amount": "1", "account": "Checking", "currency": "USD", "date": "2026-03-10"}),
        ],
        minimum_gap_days=30,
    )

    assert rows(text) == [
        {
            "account": "Checking",
            "currency": "USD",
            "previous_date": "2026-01-20",
            "next_date": "2026-03-01",
            "gap_days": "40",
            "previous_unit_id": "usd-2",
            "next_unit_id": "usd-3",
            "source": "Bank; Bank B",
        }
    ]


def test_transaction_gap_uses_unknown_account_and_writes_stats(tmp_path):
    units = [
        {"id": "a", "source_project": "Cash", "metadata": {"amount": "1", "currency": "USD", "date": "2026-01-01"}},
        {"id": "b", "source_project": "Cash", "metadata": {"amount": "1", "currency": "USD", "date": "2026-02-05"}},
        {"id": "c", "source_project": "Cash", "metadata": {"amount": "bad", "currency": "USD", "date": "2026-05-01"}},
        {"id": "d", "source_project": "Cash", "metadata": {"amount": "1", "currency": "USD"}},
    ]
    expected = export_units_to_transaction_gap_csv(units, minimum_gap_days=10)
    path = tmp_path / "transaction-gap.csv"

    stats = export_units_to_transaction_gap_csv(units, path, minimum_gap_days=10)

    assert rows(expected) == [
        {
            "account": "Unknown",
            "currency": "USD",
            "previous_date": "2026-01-01",
            "next_date": "2026-02-05",
            "gap_days": "35",
            "previous_unit_id": "a",
            "next_unit_id": "b",
            "source": "Cash",
        }
    ]
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 4
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size


def test_transaction_gap_validates_minimum_gap_days():
    with pytest.raises(ValueError):
        export_units_to_transaction_gap_csv([], minimum_gap_days=-1)
