from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_balance_timeline_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, title: str | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="Bank",
        source_id=f"source-{unit_id}",
        source_entity_type="transaction",
        title=title or f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_balance_timeline_includes_parseable_balances_and_deltas():
    text = export_unit_balance_timeline_csv(
        [
            unit("b", metadata={"balance": "$1,050.50", "amount": "+50.50", "currency": "USD", "account": "Checking", "date": "2026-05-02"}),
            unit("a", metadata={"ending_balance": "$1,000.00", "amount": "($10.00)", "currency": "USD", "account": "Checking", "date": "2026-05-01"}),
            unit("c", metadata={"balance": "bad", "currency": "USD"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Bank",
            "account": "Checking",
            "currency": "USD",
            "date": "2026-05-01",
            "balance": "1000",
            "amount": "-10",
            "delta_from_previous": "",
            "unit_id": "a",
            "title": "Title a",
        },
        {
            "source_project": "Bank",
            "account": "Checking",
            "currency": "USD",
            "date": "2026-05-02",
            "balance": "1050.5",
            "amount": "50.5",
            "delta_from_previous": "50.5",
            "unit_id": "b",
            "title": "Title b",
        },
    ]


def test_balance_timeline_unknowns_and_path_stats(tmp_path):
    units = [{"id": "a", "source_project": "Cash", "title": "Cash row", "metadata": {"balance": "10"}}]
    expected = export_unit_balance_timeline_csv(units)
    path = tmp_path / "timeline.csv"

    stats = export_unit_balance_timeline_csv(units, path)

    row = rows(expected)[0]
    assert row["account"] == "Unknown"
    assert row["currency"] == "Unknown"
    assert row["date"] == "Unknown"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
