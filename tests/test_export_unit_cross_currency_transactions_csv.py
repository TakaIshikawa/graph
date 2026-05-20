from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_cross_currency_transactions_csv import export_units_to_cross_currency_transactions_csv
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


def test_cross_currency_transactions_emit_rows_for_distinct_alternate_currencies():
    text = export_units_to_cross_currency_transactions_csv(
        [
            unit(
                "a",
                metadata={
                    "amount": "110.00",
                    "currency": "USD",
                    "original_amount": "100.00",
                    "original_currency": "eur",
                    "fee_amount": "2.00",
                    "fee_currency": "USD",
                    "exchange_rate": "1.1",
                    "date": "2026-01-01",
                },
                source_project="Card",
            ),
            unit(
                "b",
                metadata={
                    "amount": "150",
                    "currency": "USD",
                    "settlement_amount": "200",
                    "settlement_currency": "CAD",
                    "converted_amount": "150",
                    "converted_currency": "USD",
                    "fx_rate": "0.75",
                    "date": "2026-01-02",
                },
            ),
            unit("same", metadata={"amount": "5", "currency": "USD", "settlement_currency": "usd", "date": "2026-01-03"}),
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "date": "2026-01-01",
            "amount": "110",
            "currency": "USD",
            "alternate_currency": "EUR",
            "alternate_amount": "100.00",
            "exchange_rate": "1.1",
            "source": "Card",
            "evidence": "original_currency=EUR",
        },
        {
            "unit_id": "b",
            "date": "2026-01-02",
            "amount": "150",
            "currency": "USD",
            "alternate_currency": "CAD",
            "alternate_amount": "200",
            "exchange_rate": "0.75",
            "source": "Bank",
            "evidence": "settlement_currency=CAD",
        },
    ]


def test_cross_currency_transactions_leave_missing_exchange_rate_blank_and_write_stats(tmp_path):
    units = [
        {"id": "a", "source_project": "Wallet", "metadata": {"amount": "10", "currency": "GBP", "source_amount": "12", "source_currency": "USD", "date": "2026-01-01"}},
        {"id": "b", "source_project": "Wallet", "metadata": {"amount": "10", "currency": "GBP"}},
    ]
    expected = export_units_to_cross_currency_transactions_csv(units)
    path = tmp_path / "cross-currency.csv"

    stats = export_units_to_cross_currency_transactions_csv(units, path)

    assert rows(expected) == [
        {
            "unit_id": "a",
            "date": "2026-01-01",
            "amount": "10",
            "currency": "GBP",
            "alternate_currency": "USD",
            "alternate_amount": "12",
            "exchange_rate": "",
            "source": "Wallet",
            "evidence": "source_currency=USD",
        }
    ]
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 2
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
