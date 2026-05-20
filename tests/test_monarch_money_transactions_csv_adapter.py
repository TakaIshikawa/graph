from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.monarch_money_transactions_csv import MonarchMoneyTransactionsCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.models import SyncState


def test_monarch_money_transactions_csv_ingests_rows(tmp_path):
    export = tmp_path / "monarch.csv"
    export.write_text(
        "Date,Transaction ID,Merchant,Original Merchant,Account,Category,Amount,Currency,Notes,Tags\n"
        "2026-05-01,mm-1,Blue Bottle,BLUE BOTTLE,Checking,Coffee,-5.75,USD,Latte,\"coffee,work\"\n",
        encoding="utf-8",
    )

    result = MonarchMoneyTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "monarch_money_transactions_csv"
    assert unit.source_id == "monarch_money_transactions_csv:mm-1"
    assert unit.source_entity_type == "transaction"
    assert unit.metadata["timestamp"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["merchant"] == "Blue Bottle"
    assert unit.metadata["payee"] == "Blue Bottle"
    assert unit.metadata["original_merchant"] == "BLUE BOTTLE"
    assert unit.metadata["account"] == "Checking"
    assert unit.metadata["category"] == "Coffee"
    assert unit.metadata["amount"] == -5.75
    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["notes"] == "Latte"
    assert unit.metadata["tags"] == ["coffee", "work"]
    assert unit.metadata["source_file"] == "monarch.csv"
    assert unit.metadata["source_row"]["Transaction ID"] == "mm-1"


def test_monarch_money_transactions_csv_handles_directory_since_and_filters(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "new.csv").write_text("Date,Merchant,Account,Amount\n2026-05-03,New,Checking,-2\n", encoding="utf-8")
    (export_dir / "old.csv").write_text("Date,Merchant,Account,Amount\n2026-05-01,Old,Checking,-1\n", encoding="utf-8")
    (export_dir / "blank.csv").write_text("\n", encoding="utf-8")
    (export_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    adapter = MonarchMoneyTransactionsCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="monarch_money_transactions_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    assert [unit.metadata["merchant"] for unit in adapter.ingest(since=since).units] == ["New"]
    assert adapter.ingest(entity_types=["note"]).units == []


def test_monarch_money_transactions_csv_source_ids_are_stable_without_transaction_id(tmp_path):
    export = tmp_path / "monarch.csv"
    export.write_text(
        "Date,Merchant,Account,Category,Amount\n"
        "2026-05-01,Coffee,Checking,Food,-4.50\n",
        encoding="utf-8",
    )

    first = MonarchMoneyTransactionsCsvAdapter(path=str(export)).ingest().units
    second = MonarchMoneyTransactionsCsvAdapter(path=str(export)).ingest().units

    assert [unit.source_id for unit in first] == [unit.source_id for unit in second]
    assert first[0].source_id.startswith("monarch_money_transactions_csv:")


def test_monarch_money_transactions_csv_is_registered():
    assert "monarch_money_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("monarch-money-transactions-csv"), MonarchMoneyTransactionsCsvAdapter)
