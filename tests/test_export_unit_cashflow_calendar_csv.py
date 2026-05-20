from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_cashflow_calendar_csv import export_unit_cashflow_calendar_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project="Bank", source_id=f"source-{unit_id}", source_entity_type="transaction", title=unit_id, content="", metadata=metadata or {}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_cashflow_calendar_groups_daily_cashflow_and_running_delta_by_series():
    text = export_unit_cashflow_calendar_csv(
        [
            unit("b", metadata={"amount": "-3", "currency": "USD", "account": "Checking", "date": "2026-05-02"}),
            unit("a", metadata={"amount": "10", "currency": "USD", "account": "Checking", "date": "2026-05-01"}),
            unit("c", metadata={"amount": "-2", "currency": "USD", "account": "Checking", "date": "2026-05-02"}),
            unit("d", metadata={"amount": "7", "currency": "EUR", "account": "Checking", "date": "2026-05-01"}),
        ]
    )

    assert rows(text) == [
        {"date": "2026-05-01", "source_project": "Bank", "account": "Checking", "currency": "EUR", "transaction_count": "1", "debit_total": "0", "credit_total": "7", "net_amount": "7", "running_balance_delta": "7"},
        {"date": "2026-05-01", "source_project": "Bank", "account": "Checking", "currency": "USD", "transaction_count": "1", "debit_total": "0", "credit_total": "10", "net_amount": "10", "running_balance_delta": "10"},
        {"date": "2026-05-02", "source_project": "Bank", "account": "Checking", "currency": "USD", "transaction_count": "2", "debit_total": "-5", "credit_total": "0", "net_amount": "-5", "running_balance_delta": "5"},
    ]


def test_cashflow_calendar_skips_missing_date_or_amount_and_writes_stats(tmp_path):
    units = [
        {"id": "a", "source_project": "Cash", "metadata": {"amount": "1", "date": "2026-05-20"}},
        {"id": "b", "source_project": "Cash", "metadata": {"amount": "2"}},
        {"id": "c", "source_project": "Cash", "metadata": {"date": "2026-05-20"}},
    ]
    expected = export_unit_cashflow_calendar_csv(units)
    path = tmp_path / "calendar.csv"

    stats = export_unit_cashflow_calendar_csv(units, path)

    row = rows(expected)[0]
    assert row["account"] == "Unknown"
    assert row["currency"] == "Unknown"
    assert row["running_balance_delta"] == "1"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 3
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
