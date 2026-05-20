from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_tax_year_summary_csv import export_units_to_tax_year_summary_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project="Bank", source_id=f"source-{unit_id}", source_entity_type="transaction", title=unit_id, content="", metadata=metadata or {}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_tax_year_summary_groups_by_year_and_currency_in_deterministic_order():
    text = export_units_to_tax_year_summary_csv(
        [
            unit("b", metadata={"amount": "($5.50)", "currency": "USD", "posted_date": "2026-05-03"}),
            unit("a", metadata={"transaction_amount": "$10.00", "currency": "USD", "transaction_date": "2026-01-01"}),
            unit("c", metadata={"amount": "7", "currency": "EUR", "date": "2026-05-02"}),
            unit("d", metadata={"amount": "4", "currency": "USD", "date": "2025-12-31"}),
        ]
    )

    assert rows(text) == [
        {"year": "2025", "currency": "USD", "transaction_count": "1", "debit_total": "0", "credit_total": "4", "net_total": "4", "first_date": "2025-12-31", "last_date": "2025-12-31"},
        {"year": "2026", "currency": "EUR", "transaction_count": "1", "debit_total": "0", "credit_total": "7", "net_total": "7", "first_date": "2026-05-02", "last_date": "2026-05-02"},
        {"year": "2026", "currency": "USD", "transaction_count": "2", "debit_total": "-5.5", "credit_total": "10", "net_total": "4.5", "first_date": "2026-01-01", "last_date": "2026-05-03"},
    ]


def test_tax_year_summary_skips_invalid_rows_and_writes_stats(tmp_path):
    units = [
        {"id": "a", "source_project": "Cash", "metadata": {"amount": "+1,200.25", "date": "2026-05-01"}},
        {"id": "b", "source_project": "Cash", "metadata": {"amount": "99"}},
        {"id": "c", "source_project": "Cash", "metadata": {"date": "2026-05-02"}},
        {"id": "d", "source_project": "Cash", "metadata": {"amount": "bad", "date": "2026-05-03"}},
    ]
    expected = export_units_to_tax_year_summary_csv(units)
    path = tmp_path / "tax-year.csv"

    stats = export_units_to_tax_year_summary_csv(units, path)

    assert rows(expected) == [{"year": "2026", "currency": "Unknown", "transaction_count": "1", "debit_total": "0", "credit_total": "1200.25", "net_total": "1200.25", "first_date": "2026-05-01", "last_date": "2026-05-01"}]
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 4
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
