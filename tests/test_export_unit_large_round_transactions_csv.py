from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export.unit_large_round_transactions_csv import export_units_to_large_round_transactions_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="Bank",
        source_id=f"source-{unit_id}",
        source_entity_type="transaction",
        title=unit_id,
        content="",
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_large_round_transactions_filters_by_default_threshold_and_increment_and_sorts():
    text = export_units_to_large_round_transactions_csv(
        [
            unit("small", metadata={"amount": "900", "currency": "USD", "account": "Checking", "date": "2026-01-01"}),
            unit("not-round", metadata={"amount": "1050", "currency": "USD", "account": "Checking", "date": "2026-01-01"}),
            unit("debit", metadata={"amount": "-2000", "currency": "USD", "account": "Checking", "merchant": "Rent", "description": "Rent payment", "date": "2026-01-02"}),
            unit("credit", metadata={"amount": "1000", "currency": "USD", "account": "Checking", "payee": "Payroll", "memo": "Salary", "date": "2026-01-02"}),
            unit("early", metadata={"amount": "3000", "currency": "EUR", "account_name": "Savings", "counterparty": "Broker", "note": "Transfer", "date": "2026-01-01"}),
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "early",
            "date": "2026-01-01",
            "amount": "3000",
            "currency": "EUR",
            "account": "Savings",
            "counterparty": "Broker",
            "description": "Transfer",
            "round_increment": "100",
        },
        {
            "unit_id": "debit",
            "date": "2026-01-02",
            "amount": "-2000",
            "currency": "USD",
            "account": "Checking",
            "counterparty": "Rent",
            "description": "Rent payment",
            "round_increment": "100",
        },
        {
            "unit_id": "credit",
            "date": "2026-01-02",
            "amount": "1000",
            "currency": "USD",
            "account": "Checking",
            "counterparty": "Payroll",
            "description": "Salary",
            "round_increment": "100",
        },
    ]


def test_large_round_transactions_accepts_configurable_threshold_increment_and_path(tmp_path):
    units = [
        {"id": "a", "metadata": {"amount": "250", "currency": "USD", "date": "2026-01-01"}},
        {"id": "b", "metadata": {"amount": "300", "currency": "USD", "date": "2026-01-02"}},
        {"id": "c", "metadata": {"amount": "375", "currency": "USD", "date": "2026-01-03"}},
    ]
    expected = export_units_to_large_round_transactions_csv(units, minimum_abs_amount="250", round_increment="25")
    path = tmp_path / "large-round.csv"

    stats = export_units_to_large_round_transactions_csv(units, path, minimum_abs_amount="250", round_increment="25")

    assert [row["unit_id"] for row in rows(expected)] == ["a", "b", "c"]
    assert {row["round_increment"] for row in rows(expected)} == {"25"}
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 3
    assert stats["rows_exported"] == 3
    assert stats["bytes_written"] == path.stat().st_size


@pytest.mark.parametrize("kwargs", [{"minimum_abs_amount": "0"}, {"round_increment": "-1"}])
def test_large_round_transactions_validates_positive_configuration(kwargs: dict[str, str]):
    with pytest.raises(ValueError):
        export_units_to_large_round_transactions_csv([], **kwargs)
