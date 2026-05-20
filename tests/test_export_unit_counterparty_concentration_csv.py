from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_counterparty_concentration_csv import export_unit_counterparty_concentration_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, source_project: object = "Bank") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project=source_project, source_id=f"source-{unit_id}", source_entity_type="transaction", title=unit_id, content="", metadata=metadata or {}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_counterparty_concentration_groups_normalized_counterparties_and_shares():
    text = export_unit_counterparty_concentration_csv(
        [
            unit("a", metadata={"merchant": "Cafe", "amount": "-10", "currency": "USD", "account": "Card", "date": "2026-05-01"}),
            unit("b", metadata={"payee": "cafe", "amount": "5", "currency": "USD", "account": "Card", "date": "2026-05-03"}),
            unit("c", metadata={"counterparty": "Book Shop", "amount": "-15", "currency": "USD", "account": "Card", "date": "2026-05-02"}),
        ]
    )

    assert rows(text) == [
        {"source_project": "Bank", "account": "Card", "currency": "USD", "counterparty": "book shop", "total_amount_abs": "15", "debit_total": "-15", "credit_total": "0", "transaction_count": "1", "share_of_source_amount": "0.5", "first_seen": "2026-05-02", "last_seen": "2026-05-02"},
        {"source_project": "Bank", "account": "Card", "currency": "USD", "counterparty": "cafe", "total_amount_abs": "15", "debit_total": "-10", "credit_total": "5", "transaction_count": "2", "share_of_source_amount": "0.5", "first_seen": "2026-05-01", "last_seen": "2026-05-03"},
    ]


def test_counterparty_concentration_unknown_counterparty_and_path_stats(tmp_path):
    units = [{"id": "a", "source_project": "Cash", "metadata": {"amount": "1"}}]
    expected = export_unit_counterparty_concentration_csv(units)
    path = tmp_path / "counterparties.csv"

    stats = export_unit_counterparty_concentration_csv(units, path)

    row = rows(expected)[0]
    assert row["counterparty"] == "Unknown"
    assert row["account"] == "Unknown"
    assert row["currency"] == "Unknown"
    assert row["share_of_source_amount"] == "1"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
