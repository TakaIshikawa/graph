from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_duplicate_transactions_csv import export_unit_duplicate_transactions_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, source_project: object = "Bank") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project=source_project, source_id=f"source-{unit_id}", source_entity_type="transaction", title=unit_id, content="", metadata=metadata or {}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_duplicate_transactions_emit_one_row_per_matching_pair():
    text = export_unit_duplicate_transactions_csv(
        [
            unit("b", metadata={"amount": "-12.34", "currency": "USD", "merchant": "Cafe", "account": "Card", "date": "2026-05-01"}),
            unit("a", metadata={"amount": "($12.34)", "currency": "USD", "payee": "cafe", "account": "Card", "date": "2026-05-01"}),
            unit("c", metadata={"amount": "-12.34", "currency": "USD", "merchant": "Cafe", "account": "Card", "date": "2026-05-02"}),
        ]
    )

    assert rows(text) == [
        {"left_unit_id": "a", "right_unit_id": "b", "amount": "-12.34", "currency": "USD", "date": "2026-05-01", "counterparty": "cafe", "same_source": "true", "same_account": "true", "evidence": "same date, amount, currency, and counterparty"}
    ]


def test_duplicate_transactions_compare_source_account_and_write_stats(tmp_path):
    units = [
        {"id": "a", "source_project": "A", "metadata": {"amount": "20", "currency": "USD", "counterparty": "Shop", "account": "One", "date": "2026-05-01"}},
        {"id": "b", "source_project": "B", "metadata": {"amount": "20", "currency": "USD", "counterparty": "Shop", "account": "Two", "date": "2026-05-01"}},
        {"id": "skip", "source_project": "B", "metadata": {"amount": "20", "currency": "USD", "counterparty": "Shop"}},
    ]
    expected = export_unit_duplicate_transactions_csv(units)
    path = tmp_path / "duplicates.csv"

    stats = export_unit_duplicate_transactions_csv(units, path)

    row = rows(expected)[0]
    assert row["same_source"] == "false"
    assert row["same_account"] == "false"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 3
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
