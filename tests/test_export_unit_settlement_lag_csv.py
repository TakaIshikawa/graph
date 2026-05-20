from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_settlement_lag_csv import export_unit_settlement_lag_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project="Broker", source_id=f"source-{unit_id}", source_entity_type="transaction", title=unit_id, content="", metadata=metadata or {}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_settlement_lag_groups_lag_statistics_by_source_account_currency():
    text = export_unit_settlement_lag_csv(
        [
            unit("a", metadata={"trade_date": "2026-05-01", "settlement_date": "2026-05-03", "currency": "USD", "account": "Taxable"}),
            unit("b", metadata={"transaction_date": "2026-05-02", "posted_date": "2026-05-06", "currency": "USD", "account": "Taxable"}),
            unit("c", metadata={"trade_date": "2026-05-01", "currency": "USD", "account": "Taxable"}),
        ]
    )

    assert rows(text) == [
        {"source_project": "Broker", "account": "Taxable", "currency": "USD", "count": "2", "average_lag_days": "3", "median_lag_days": "3", "max_lag_days": "4", "representative_unit_ids": "a; b"}
    ]


def test_settlement_lag_supports_posted_at_unknowns_and_path_stats(tmp_path):
    units = [{"id": "a", "source_project": "Bank", "metadata": {"date": "2026-05-01", "posted_at": "2026-05-02T09:00:00", "currency": "USD"}}]
    expected = export_unit_settlement_lag_csv(units)
    path = tmp_path / "settlement.csv"

    stats = export_unit_settlement_lag_csv(units, path)

    row = rows(expected)[0]
    assert row["account"] == "Unknown"
    assert row["average_lag_days"] == "1"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
