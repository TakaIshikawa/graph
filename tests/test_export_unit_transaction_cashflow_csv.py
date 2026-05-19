from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_transaction_cashflow_csv
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


def test_transaction_cashflow_groups_inflow_outflow_and_net_amounts():
    text = export_unit_transaction_cashflow_csv(
        [
            unit("b", metadata={"amount": "($5.50)", "currency": "USD", "account": "Checking", "date": "2026-05-03"}),
            unit("a", metadata={"transaction_amount": "$10.00", "currency": "USD", "account": "Checking", "date": "2026-05-01"}),
            unit("c", metadata={"change": "bad", "currency": "USD"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Bank",
            "account": "Checking",
            "currency": "USD",
            "month": "2026-05",
            "transaction_count": "2",
            "inflow_total": "10",
            "outflow_total": "-5.5",
            "net_amount": "4.5",
            "representative_unit_ids": "a; b",
        }
    ]


def test_transaction_cashflow_parses_commas_signs_and_unknowns_from_mapping():
    text = export_unit_transaction_cashflow_csv(
        [{"id": "a", "source_project": "Cash", "metadata": {"amount": "+1,200.25"}}]
    )

    row = rows(text)[0]
    assert row["account"] == "Unknown"
    assert row["currency"] == "Unknown"
    assert row["month"] == "Unknown"
    assert row["inflow_total"] == "1200.25"
    assert row["outflow_total"] == "0"
    assert row["net_amount"] == "1200.25"


def test_transaction_cashflow_path_and_file_like_writes_stats(tmp_path):
    units = [unit("a", metadata={"amount": "1"})]
    expected = export_unit_transaction_cashflow_csv(units)
    path = tmp_path / "reports" / "cashflow.csv"

    stats = export_unit_transaction_cashflow_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size

    output = StringIO()
    stream_stats = export_unit_transaction_cashflow_csv(units, output)
    assert output.getvalue() == expected
    assert stream_stats["unit_count"] == 1
    assert stream_stats["rows_exported"] == 1
    assert stream_stats["bytes_written"] == len(expected)
