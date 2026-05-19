from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_currency_coverage_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, source_project: object = "Bank", entity_type: str = "transaction") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=entity_type,
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_currency_coverage_groups_and_counts_currency_usage():
    text = export_source_currency_coverage_csv(
        [
            unit("a", metadata={"currency": " usd "}),
            unit("b", metadata={"asset": "btc"}),
            unit("c", metadata={"base_currency": "usd", "quote_currency": "eur"}),
            unit("d", metadata={}),
            unit("e", metadata={"currency": "JPY"}, source_project="Broker"),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Bank",
            "source_entity_type": "transaction",
            "unit_count": "4",
            "unique_currency_count": "3",
            "currencies": "BTC; EUR; USD",
            "missing_currency_count": "1",
            "dominant_currency": "USD",
            "representative_unit_ids": "a; b; c; d",
        },
        {
            "source_project": "Broker",
            "source_entity_type": "transaction",
            "unit_count": "1",
            "unique_currency_count": "1",
            "currencies": "JPY",
            "missing_currency_count": "0",
            "dominant_currency": "JPY",
            "representative_unit_ids": "e",
        },
    ]


def test_source_currency_coverage_accepts_mapping_and_writes_stats(tmp_path):
    units = [{"id": "a", "source_project": "Cash", "source_entity_type": "payment", "metadata": {"transaction_currency": " usd "}}]
    expected = export_source_currency_coverage_csv(units)
    path = tmp_path / "coverage.csv"

    stats = export_source_currency_coverage_csv(units, path)

    assert rows(expected)[0]["currencies"] == "USD"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
