from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.rocket_money_transactions_csv import RocketMoneyTransactionsCsvAdapter
from graph.types.models import SyncState


def test_rocket_money_transactions_csv_ingests_rows(tmp_path):
    export = tmp_path / "rocket.csv"
    export.write_text(
        "Date,Transaction ID,Description,Account,Category,Amount,Currency,Status,Notes\n"
        "2026-05-01,rm-1,Coffee,Checking,Food,-4.25,USD,Pending,Cold brew\n",
        encoding="utf-8",
    )

    unit = RocketMoneyTransactionsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "rocket_money_transactions_csv"
    assert unit.source_id == "rocket_money_transactions_csv:rm-1"
    assert unit.metadata["timestamp"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["description"] == "Coffee"
    assert unit.metadata["merchant"] == "Coffee"
    assert unit.metadata["account"] == "Checking"
    assert unit.metadata["category"] == "Food"
    assert unit.metadata["amount"] == -4.25
    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["status"] == "Pending"
    assert unit.metadata["pending"] is True
    assert unit.metadata["notes"] == "Cold brew"
    assert unit.metadata["source_file"] == "rocket.csv"
    assert unit.metadata["source_row"]["Transaction ID"] == "rm-1"


def test_rocket_money_transactions_csv_directory_since_filters_and_stable_ids(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Date,Description,Amount,Status\n2026-05-01,Old,-1,Cleared\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("Date,Description,Amount,Status\n2026-05-03,New,-2,Posted\n", encoding="utf-8")
    (export_dir / "blank.csv").write_text("\n", encoding="utf-8")

    adapter = RocketMoneyTransactionsCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="rocket_money_transactions_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    assert [unit.metadata["description"] for unit in adapter.ingest(since=since).units] == ["New"]
    assert adapter.ingest(entity_types=["note"]).units == []
    assert [unit.source_id for unit in adapter.ingest().units] == [unit.source_id for unit in adapter.ingest().units]


def test_rocket_money_transactions_csv_is_registered():
    assert "rocket_money_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("rocket-money-transactions-csv"), RocketMoneyTransactionsCsvAdapter)
