from __future__ import annotations

import csv
from datetime import date
from io import StringIO

from graph.export import export_source_account_summary_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, source_project: object = "Bank", entity_type: str = "transaction", created_at: object = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=entity_type,
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[],
        created_at=created_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_account_summary_groups_accounts_and_totals_amounts():
    text = export_source_account_summary_csv(
        [
            unit("b", metadata={"account_name": "Checking", "institution": "Local Bank", "currency": "USD", "amount": "($5.50)", "date": "2026-05-03"}),
            unit("a", metadata={"account": "Checking", "institution": "Local Bank", "currency": "USD", "transaction_amount": "$10.00", "date": "2026-05-01"}),
            unit("c", metadata={"account_id": "Brokerage", "brokerage": "Broker", "currency": "USD", "amount": "bad", "source_date": "2026-05-02"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Bank",
            "institution": "Broker",
            "account": "Brokerage",
            "source_entity_type": "transaction",
            "unit_count": "1",
            "first_seen": "2026-05-02",
            "last_seen": "2026-05-02",
            "currency": "USD",
            "total_amount": "",
            "representative_unit_ids": "c",
        },
        {
            "source_project": "Bank",
            "institution": "Local Bank",
            "account": "Checking",
            "source_entity_type": "transaction",
            "unit_count": "2",
            "first_seen": "2026-05-01",
            "last_seen": "2026-05-03",
            "currency": "USD",
            "total_amount": "4.5",
            "representative_unit_ids": "a; b",
        },
    ]


def test_source_account_summary_uses_unknown_for_missing_account_and_institution_and_unit_dates():
    text = export_source_account_summary_csv([unit("a", metadata={}, created_at=date(2026, 5, 1))])

    assert rows(text)[0] == {
        "source_project": "Bank",
        "institution": "Unknown",
        "account": "Unknown",
        "source_entity_type": "transaction",
        "unit_count": "1",
        "first_seen": "2026-05-01",
        "last_seen": "2026-05-01",
        "currency": "",
        "total_amount": "",
        "representative_unit_ids": "a",
    }


def test_source_account_summary_supports_mapping_inputs_and_path_writes(tmp_path):
    units = [{"id": "a", "metadata": {"account": "Cash", "amount": "1,200.00"}}]
    expected = export_source_account_summary_csv(units)
    path = tmp_path / "accounts.csv"

    stats = export_source_account_summary_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
