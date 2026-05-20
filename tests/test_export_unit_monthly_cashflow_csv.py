from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_monthly_cashflow_csv import export_unit_monthly_cashflow_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, source_project: object = "Bank") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project=source_project, source_id=f"source-{unit_id}", source_entity_type="transaction", title=unit_id, content="", metadata=metadata or {}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_monthly_cashflow_groups_by_month_source_account_and_currency():
    text = export_unit_monthly_cashflow_csv(
        [
            unit("b", metadata={"amount": "($5.50)", "currency": "USD", "account": "Checking", "date": "2026-05-03"}),
            unit("a", metadata={"transaction_amount": "$10.00", "currency": "USD", "account": "Checking", "date": "2026-05-01"}),
            unit("c", metadata={"amount": "7", "currency": "EUR", "account": "Checking", "date": "2026-05-02"}),
            unit("d", metadata={"amount": "4", "currency": "USD", "account": "Checking", "date": "2026-06-01"}),
        ]
    )

    assert rows(text) == [
        {"year_month": "2026-05", "source_project": "Bank", "account": "Checking", "currency": "EUR", "transaction_count": "1", "debit_total": "0", "credit_total": "7", "net_amount": "7", "first_date": "2026-05-02", "last_date": "2026-05-02"},
        {"year_month": "2026-05", "source_project": "Bank", "account": "Checking", "currency": "USD", "transaction_count": "2", "debit_total": "-5.5", "credit_total": "10", "net_amount": "4.5", "first_date": "2026-05-01", "last_date": "2026-05-03"},
        {"year_month": "2026-06", "source_project": "Bank", "account": "Checking", "currency": "USD", "transaction_count": "1", "debit_total": "0", "credit_total": "4", "net_amount": "4", "first_date": "2026-06-01", "last_date": "2026-06-01"},
    ]


def test_monthly_cashflow_skips_missing_amount_or_date_and_writes_stats(tmp_path):
    units = [
        {"id": "a", "source_project": "Cash", "metadata": {"amount": "+1,200.25", "date": "2026-05-01"}},
        {"id": "b", "source_project": "Cash", "metadata": {"amount": "99"}},
        {"id": "c", "source_project": "Cash", "metadata": {"date": "2026-05-02"}},
    ]
    expected = export_unit_monthly_cashflow_csv(units)
    path = tmp_path / "monthly.csv"

    stats = export_unit_monthly_cashflow_csv(units, path)

    row = rows(expected)[0]
    assert row["account"] == "Unknown"
    assert row["currency"] == "Unknown"
    assert row["credit_total"] == "1200.25"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 3
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
