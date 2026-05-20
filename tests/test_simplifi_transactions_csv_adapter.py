from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.simplifi_transactions_csv import SimplifiTransactionsCsvAdapter
from graph.types.models import SyncState


def test_simplifi_transactions_csv_ingests_rows(tmp_path):
    export = tmp_path / "simplifi.csv"
    export.write_text(
        "Date,Transaction ID,Payee,Category,Tags,Account,Amount,Notes,Status,Type\n"
        "2026-05-01,sf-1,Grocery Store,Groceries,\"food,home\",Checking,-82.10,Weekly shop,Cleared,Expense\n",
        encoding="utf-8",
    )

    unit = SimplifiTransactionsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "simplifi_transactions_csv"
    assert unit.source_id == "simplifi_transactions_csv:sf-1"
    assert unit.metadata["timestamp"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["payee"] == "Grocery Store"
    assert unit.metadata["category"] == "Groceries"
    assert unit.metadata["tags"] == ["food", "home"]
    assert unit.metadata["account"] == "Checking"
    assert unit.metadata["amount"] == -82.10
    assert unit.metadata["notes"] == "Weekly shop"
    assert unit.metadata["status"] == "Cleared"
    assert unit.metadata["type"] == "Expense"
    assert unit.metadata["source_row"]["Transaction ID"] == "sf-1"


def test_simplifi_transactions_csv_directory_filters_and_digest_ids(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Date,Payee,Amount,Account\n2026-05-01,Old,-1,Checking\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("Date,Payee,Amount,Account\n2026-05-03,New,-2,Checking\n", encoding="utf-8")
    (export_dir / "empty.csv").write_text("\n", encoding="utf-8")

    adapter = SimplifiTransactionsCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="simplifi_transactions_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    assert [unit.metadata["payee"] for unit in adapter.ingest(since=since).units] == ["New"]
    assert adapter.ingest(entity_types=["note"]).units == []
    assert [unit.source_id for unit in adapter.ingest().units] == [unit.source_id for unit in adapter.ingest().units]


def test_simplifi_transactions_csv_is_registered():
    assert "simplifi_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("simplifi-transactions-csv"), SimplifiTransactionsCsvAdapter)
