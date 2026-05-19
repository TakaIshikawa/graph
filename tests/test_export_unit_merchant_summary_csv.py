from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_merchant_summary_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="Bank",
        source_id=f"source-{unit_id}",
        source_entity_type="transaction",
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_merchant_summary_groups_totals_and_seen_dates():
    text = export_unit_merchant_summary_csv(
        [
            unit("b", metadata={"payee": "Cafe", "amount": "($5.00)", "currency": "USD", "account": "Card", "category": "Food", "date": "2026-05-03"}),
            unit("a", metadata={"merchant": "Cafe", "amount": "$10.00", "currency": "USD", "account": "Card", "category": "Food", "date": "2026-05-01"}),
            unit("c", metadata={"merchant": "Cafe", "amount": "bad"}),
        ]
    )

    assert rows(text) == [
        {
            "merchant": "Cafe",
            "source_project": "Bank",
            "account": "Card",
            "category": "Food",
            "currency": "USD",
            "transaction_count": "2",
            "total_amount": "5",
            "debit_total": "-5",
            "credit_total": "10",
            "first_seen": "2026-05-01",
            "last_seen": "2026-05-03",
            "average_amount": "2.5",
            "representative_unit_ids": "a; b",
        }
    ]


def test_merchant_summary_unknowns_mapping_and_path_stats(tmp_path):
    units = [{"id": "a", "source_project": "Cash", "metadata": {"amount": "1"}}]
    expected = export_unit_merchant_summary_csv(units)
    path = tmp_path / "merchant.csv"

    stats = export_unit_merchant_summary_csv(units, path)

    row = rows(expected)[0]
    assert row["merchant"] == "Unknown"
    assert row["account"] == "Unknown"
    assert row["category"] == "Unknown"
    assert row["currency"] == "Unknown"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
