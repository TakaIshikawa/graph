from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_refund_candidates_csv import export_unit_refund_candidates_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project="Bank", source_id=f"source-{unit_id}", source_entity_type="transaction", title=unit_id, content="", metadata=metadata or {}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_refund_candidates_match_opposite_sign_same_merchant_within_window():
    text = export_unit_refund_candidates_csv(
        [
            unit("purchase", metadata={"amount": "($20.00)", "currency": "USD", "merchant": "Store", "date": "2026-05-01"}),
            unit("refund", metadata={"amount": "$18.00", "currency": "USD", "payee": "store", "date": "2026-05-04"}),
            unit("other", metadata={"amount": "$20.00", "currency": "USD", "merchant": "Other", "date": "2026-05-02"}),
        ],
        max_days=10,
    )

    assert rows(text) == [
        {"debit_unit_id": "purchase", "credit_unit_id": "refund", "merchant": "store", "currency": "USD", "debit_amount": "-20", "credit_amount": "18", "date_lag_days": "3", "amount_delta": "2", "confidence": "0.86", "evidence": "opposite signs with same merchant and currency within 3 days"}
    ]


def test_refund_candidates_allow_same_account_skip_missing_merchant_and_write_stats(tmp_path):
    units = [
        {"id": "a", "metadata": {"amount": "-10", "currency": "USD", "counterparty": "Shop", "account": "Card", "date": "2026-05-01"}},
        {"id": "b", "metadata": {"amount": "10", "currency": "USD", "counterparty": "Shop", "account": "Card", "date": "2026-05-02"}},
        {"id": "skip", "metadata": {"amount": "10", "currency": "USD", "date": "2026-05-02"}},
    ]
    expected = export_unit_refund_candidates_csv(units, max_days=3)
    path = tmp_path / "refunds.csv"

    stats = export_unit_refund_candidates_csv(units, path, max_days=3)

    assert len(rows(expected)) == 1
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 3
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
