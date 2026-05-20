from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_transaction_completeness_csv import export_unit_transaction_completeness_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, source_project: object = "Bank") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project=source_project, source_id=f"source-{unit_id}", source_entity_type="transaction", title=unit_id, content="", metadata=metadata or {}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_transaction_completeness_counts_required_field_presence_by_source_account():
    text = export_unit_transaction_completeness_csv(
        [
            unit("complete", metadata={"amount": "1", "currency": "USD", "date": "2026-05-01", "merchant": "Cafe", "category": "Food", "memo": "Latte", "account": "Card"}),
            unit("missing", metadata={"amount": "bad", "currency": "USD", "date": "bad", "account": "Card"}),
        ]
    )

    row = rows(text)[0]
    assert row["source_project"] == "Bank"
    assert row["account"] == "Card"
    assert row["total_count"] == "2"
    assert row["amount_present_count"] == "1"
    assert row["amount_missing_count"] == "1"
    assert row["amount_completeness_pct"] == "50"
    assert row["counterparty_present_count"] == "1"
    assert row["description_present_count"] == "1"
    assert row["sample_missing_unit_ids"] == "missing"


def test_transaction_completeness_recognizes_aliases_and_path_stats(tmp_path):
    units = [{"id": "a", "source_project": "Cash", "source_entity_type": "row", "metadata": {"transaction_amount": "2", "transaction_currency": "USD", "posted_date": "2026-05-01", "payee": "Shop", "type": "Retail", "description": "Order"}}]
    expected = export_unit_transaction_completeness_csv(units)
    path = tmp_path / "completeness.csv"

    stats = export_unit_transaction_completeness_csv(units, path)

    row = rows(expected)[0]
    assert row["account"] == "Unknown"
    assert row["total_count"] == "1"
    assert row["date_completeness_pct"] == "100"
    assert row["sample_missing_unit_ids"] == ""
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
