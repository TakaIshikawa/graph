from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_account_activity_summary_csv import export_units_to_account_activity_summary_csv
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


def test_account_activity_summary_groups_by_common_account_metadata_and_currency():
    text = export_units_to_account_activity_summary_csv(
        [
            unit("a", metadata={"amount": "-25.50", "account_id": "Checking", "currency": "USD", "merchant": "Cafe", "date": "2026-01-02"}),
            unit("b", metadata={"transaction_amount": "100", "account": "Checking", "currency": "USD", "payee": "Payroll", "posted_date": "2026-01-10"}),
            unit("c", metadata={"amount": "-10", "account_name": "Checking", "currency": "USD", "counterparty": " cafe ", "date": "2026-01-03"}),
            unit("d", metadata={"amount": "7", "account": "Checking", "currency": "EUR", "counterparty": "Refund", "date": "2026-01-04"}),
            unit("e", metadata={"amount": "-2", "card": "Card 1", "currency": "USD", "description": "Fee", "date": "2026-01-05"}),
        ]
    )

    assert rows(text) == [
        {
            "account": "Card 1",
            "currency": "USD",
            "transaction_count": "1",
            "debit_total": "-2",
            "credit_total": "0",
            "net_total": "-2",
            "first_activity_date": "2026-01-05",
            "last_activity_date": "2026-01-05",
            "distinct_counterparty_count": "1",
        },
        {
            "account": "Checking",
            "currency": "EUR",
            "transaction_count": "1",
            "debit_total": "0",
            "credit_total": "7",
            "net_total": "7",
            "first_activity_date": "2026-01-04",
            "last_activity_date": "2026-01-04",
            "distinct_counterparty_count": "1",
        },
        {
            "account": "Checking",
            "currency": "USD",
            "transaction_count": "3",
            "debit_total": "-35.5",
            "credit_total": "100",
            "net_total": "64.5",
            "first_activity_date": "2026-01-02",
            "last_activity_date": "2026-01-10",
            "distinct_counterparty_count": "2",
        },
    ]


def test_account_activity_summary_uses_unknown_account_and_writes_stats(tmp_path):
    units = [
        {"id": "a", "metadata": {"amount": "5", "currency": "USD", "date": "2026-01-01"}},
        {"id": "b", "metadata": {"amount": "bad", "account": "Ignored"}},
    ]
    expected = export_units_to_account_activity_summary_csv(units)
    path = tmp_path / "account-activity.csv"

    stats = export_units_to_account_activity_summary_csv(units, path)

    assert rows(expected) == [
        {
            "account": "Unknown",
            "currency": "USD",
            "transaction_count": "1",
            "debit_total": "0",
            "credit_total": "5",
            "net_total": "5",
            "first_activity_date": "2026-01-01",
            "last_activity_date": "2026-01-01",
            "distinct_counterparty_count": "0",
        }
    ]
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 2
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
