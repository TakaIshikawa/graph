from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_weekday_activity_csv import export_unit_weekday_activity_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project="Bank", source_id=f"source-{unit_id}", source_entity_type="transaction", title=unit_id, content="", metadata=metadata or {}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_weekday_activity_groups_amounts_by_weekday_source_account_currency():
    text = export_unit_weekday_activity_csv(
        [
            unit("a", metadata={"amount": "-10", "currency": "USD", "account": "Card", "date": "2026-05-18"}),
            unit("b", metadata={"amount": "5", "currency": "USD", "account": "Card", "date": "2026-05-18"}),
            unit("c", metadata={"amount": "8", "currency": "USD", "account": "Card", "date": "2026-05-19"}),
        ]
    )

    assert rows(text) == [
        {"source_project": "Bank", "account": "Card", "currency": "USD", "weekday_number": "1", "weekday_name": "Monday", "transaction_count": "2", "debit_total": "-10", "credit_total": "5", "net_amount": "-5", "average_abs_amount": "7.5"},
        {"source_project": "Bank", "account": "Card", "currency": "USD", "weekday_number": "2", "weekday_name": "Tuesday", "transaction_count": "1", "debit_total": "0", "credit_total": "8", "net_amount": "8", "average_abs_amount": "8"},
    ]


def test_weekday_activity_skips_invalid_dates_and_writes_stats(tmp_path):
    units = [
        {"id": "a", "source_project": "Cash", "metadata": {"amount": "1", "date": "2026-05-20"}},
        {"id": "b", "source_project": "Cash", "metadata": {"amount": "2", "date": "not-a-date"}},
    ]
    expected = export_unit_weekday_activity_csv(units)
    path = tmp_path / "weekday.csv"

    stats = export_unit_weekday_activity_csv(units, path)

    row = rows(expected)[0]
    assert row["account"] == "Unknown"
    assert row["currency"] == "Unknown"
    assert row["weekday_name"] == "Wednesday"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 2
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
