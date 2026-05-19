from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_fee_summary_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project="Broker", source_id=f"source-{unit_id}", source_entity_type="transaction", title=unit_id, content="", metadata=metadata or {}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_fee_summary_groups_parseable_fee_values_by_type_and_month():
    text = export_unit_fee_summary_csv(
        [
            unit("a", metadata={"commission": "($1.00)", "currency": "USD", "account": "Taxable", "date": "2026-05-01"}),
            unit("b", metadata={"commission_amount": "-2.50", "currency": "USD", "account": "Taxable", "date": "2026-05-02"}),
            unit("c", metadata={"network_fee": "$3.00", "currency": "USD", "account": "Taxable", "date": "2026-05-02"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Broker",
            "account": "Taxable",
            "currency": "USD",
            "fee_type": "Commission",
            "month": "2026-05",
            "transaction_count": "2",
            "fee_count": "2",
            "total_fee": "-3.5",
            "average_fee": "-1.75",
            "max_fee": "-1",
            "representative_unit_ids": "a; b",
        },
        {
            "source_project": "Broker",
            "account": "Taxable",
            "currency": "USD",
            "fee_type": "Network fee",
            "month": "2026-05",
            "transaction_count": "1",
            "fee_count": "1",
            "total_fee": "3",
            "average_fee": "3",
            "max_fee": "3",
            "representative_unit_ids": "c",
        },
    ]


def test_fee_summary_unknowns_mapping_and_path_stats(tmp_path):
    units = [{"id": "a", "source_project": "Cash", "metadata": {"fee": "+1,200.25"}}]
    expected = export_unit_fee_summary_csv(units)
    path = tmp_path / "fees.csv"

    stats = export_unit_fee_summary_csv(units, path)

    row = rows(expected)[0]
    assert row["account"] == "Unknown"
    assert row["currency"] == "Unknown"
    assert row["month"] == "Unknown"
    assert row["fee_type"] == "Fee"
    assert row["total_fee"] == "1200.25"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
