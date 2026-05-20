from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_income_source_summary_csv import export_units_to_income_source_summary_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, **fields: object) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=fields.pop("source_project", "Bank"),
        source_id=f"source-{unit_id}",
        source_entity_type="transaction",
        title=unit_id,
        content="",
        metadata=metadata or {},
        tags=[],
        **fields,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_income_source_summary_groups_positive_amounts_by_normalized_source_and_currency():
    text = export_units_to_income_source_summary_csv(
        [
            unit("a", metadata={"amount": "100.00", "merchant": "Acme Payroll", "currency": "USD", "date": "2026-01-02"}),
            unit("b", metadata={"transaction_amount": "$50", "payee": " acme payroll ", "currency": "USD", "posted_date": "2026-01-15"}),
            unit("c", metadata={"amount": "70", "counterparty": "Acme Payroll", "currency": "EUR", "date": "2026-01-10"}),
            unit("d", metadata={"amount": "25", "description": "Consulting Client", "currency": "USD", "date": "2026-02-01"}),
        ]
    )

    assert rows(text) == [
        {
            "source_label": "Acme Payroll",
            "currency": "EUR",
            "credit_count": "1",
            "credit_total": "70",
            "average_credit": "70",
            "first_seen": "2026-01-10",
            "last_seen": "2026-01-10",
            "representative_unit_ids": "c",
        },
        {
            "source_label": "Acme Payroll",
            "currency": "USD",
            "credit_count": "2",
            "credit_total": "150",
            "average_credit": "75",
            "first_seen": "2026-01-02",
            "last_seen": "2026-01-15",
            "representative_unit_ids": "a; b",
        },
        {
            "source_label": "Consulting Client",
            "currency": "USD",
            "credit_count": "1",
            "credit_total": "25",
            "average_credit": "25",
            "first_seen": "2026-02-01",
            "last_seen": "2026-02-01",
            "representative_unit_ids": "d",
        },
    ]


def test_income_source_summary_excludes_non_positive_and_uses_source_metadata_fallback(tmp_path):
    units = [
        {"id": "a", "source_project": "Ledger", "metadata": {"amount": "0", "merchant": "Ignored", "currency": "USD"}},
        {"id": "b", "source_project": "Ledger", "metadata": {"amount": "-10", "merchant": "Ignored", "currency": "USD"}},
        {"id": "c", "source_project": "Ledger", "source_metadata": {"source_label": "Platform Payout"}, "metadata": {"amount": "40", "currency": "USD"}},
    ]
    expected = export_units_to_income_source_summary_csv(units)
    path = tmp_path / "income-source.csv"

    stats = export_units_to_income_source_summary_csv(units, path)

    assert rows(expected) == [
        {
            "source_label": "Platform Payout",
            "currency": "USD",
            "credit_count": "1",
            "credit_total": "40",
            "average_credit": "40",
            "first_seen": "",
            "last_seen": "",
            "representative_unit_ids": "c",
        }
    ]
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 3
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
