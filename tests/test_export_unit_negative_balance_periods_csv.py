from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_negative_balance_periods_csv import export_units_to_negative_balance_periods_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project="Bank", source_id=f"source-{unit_id}", source_entity_type="transaction", title=unit_id, content="", metadata=metadata or {}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_negative_balance_periods_detect_contiguous_periods_by_account_and_currency():
    text = export_units_to_negative_balance_periods_csv(
        [
            unit("b", metadata={"balance": "-25", "currency": "USD", "account": "Checking", "date": "2026-05-02"}),
            unit("a", metadata={"balance": "10", "currency": "USD", "account": "Checking", "date": "2026-05-01"}),
            unit("c", metadata={"ending_balance": "-15", "currency": "USD", "account": "Checking", "date": "2026-05-03"}),
            unit("d", metadata={"balance": "0", "currency": "USD", "account": "Checking", "date": "2026-05-04"}),
            unit("e", metadata={"balance": "-4", "currency": "USD", "account": "Checking", "date": "2026-05-05"}),
            unit("f", metadata={"balance": "-7", "currency": "EUR", "account": "Checking", "date": "2026-05-01"}),
        ]
    )

    assert rows(text) == [
        {"account": "Checking", "currency": "EUR", "start_date": "2026-05-01", "end_date": "2026-05-01", "days": "1", "minimum_balance": "-7", "ending_balance": "-7", "sample_unit_ids": "f"},
        {"account": "Checking", "currency": "USD", "start_date": "2026-05-02", "end_date": "2026-05-03", "days": "2", "minimum_balance": "-25", "ending_balance": "-15", "sample_unit_ids": "b; c"},
        {"account": "Checking", "currency": "USD", "start_date": "2026-05-05", "end_date": "2026-05-05", "days": "1", "minimum_balance": "-4", "ending_balance": "-4", "sample_unit_ids": "e"},
    ]


def test_negative_balance_periods_skip_incomplete_rows_and_write_stats(tmp_path):
    units = [
        {"id": "a", "metadata": {"account": "Savings", "currency": "USD", "balance": "-1,200.25", "date": "2026-05-01"}},
        {"id": "b", "metadata": {"currency": "USD", "balance": "-99", "date": "2026-05-02"}},
        {"id": "c", "metadata": {"account": "Savings", "balance": "-99", "date": "2026-05-02"}},
        {"id": "d", "metadata": {"account": "Savings", "currency": "USD", "balance": "bad", "date": "2026-05-02"}},
        {"id": "e", "metadata": {"account": "Savings", "currency": "USD", "balance": "-2"}},
    ]
    expected = export_units_to_negative_balance_periods_csv(units)
    path = tmp_path / "negative-periods.csv"

    stats = export_units_to_negative_balance_periods_csv(units, path)

    assert rows(expected) == [{"account": "Savings", "currency": "USD", "start_date": "2026-05-01", "end_date": "2026-05-01", "days": "1", "minimum_balance": "-1200.25", "ending_balance": "-1200.25", "sample_unit_ids": "a"}]
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 5
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
