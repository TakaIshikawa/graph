from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_amount_summary_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, source_project: object = "Bank") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="transaction",
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_amount_summary_groups_and_totals_amounts():
    text = export_unit_amount_summary_csv(
        [
            unit("b", metadata={"amount": "($5.50)", "currency": "USD", "account": "Checking", "category": "Food", "date": "2026-05-03"}),
            unit("a", metadata={"transaction_amount": "$10.00", "currency": "USD", "account": "Checking", "category": "Food", "date": "2026-05-01"}),
            unit("c", metadata={"net_amount": "bad", "currency": "USD"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Bank",
            "currency": "USD",
            "account": "Checking",
            "category": "Food",
            "month": "2026-05",
            "unit_count": "2",
            "total_amount": "4.5",
            "debit_total": "-5.5",
            "credit_total": "10",
            "min_amount": "-5.5",
            "max_amount": "10",
            "representative_unit_ids": "a; b",
        }
    ]


def test_unit_amount_summary_parses_commas_signed_values_and_unknowns():
    text = export_unit_amount_summary_csv([{"id": "a", "source_project": "Cash", "metadata": {"amount": "+1,200.25"}}])

    row = rows(text)[0]
    assert row["currency"] == "Unknown"
    assert row["account"] == "Unknown"
    assert row["category"] == "Unknown"
    assert row["month"] == "Unknown"
    assert row["total_amount"] == "1200.25"


def test_unit_amount_summary_path_writes_stats(tmp_path):
    units = [unit("a", metadata={"amount": "1"})]
    expected = export_unit_amount_summary_csv(units)
    path = tmp_path / "reports" / "amounts.csv"

    stats = export_unit_amount_summary_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
