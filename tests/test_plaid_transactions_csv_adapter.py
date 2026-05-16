from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.plaid_transactions_csv import PlaidTransactionsCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_plaid_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "plaid.csv"
    export.write_text(
        "transaction_id,date,authorized_date,name,merchant_name,amount,iso_currency_code,account_id,account_name,category,pending\n"
        "tx1,2026-05-01,2026-04-30,Coffee purchase,Cafe,4.50,USD,acc1,Checking,\"Food, Coffee\",false\n",
        encoding="utf-8",
    )

    result = PlaidTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.PLAID_TRANSACTIONS_CSV
    assert unit.source_entity_type == "transaction"
    assert unit.source_id == "plaid_transactions_csv:tx1"
    assert unit.metadata["timestamp"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["authorized_date"] == "2026-04-30T00:00:00+00:00"
    assert unit.metadata["name"] == "Coffee purchase"
    assert unit.metadata["merchant_name"] == "Cafe"
    assert unit.metadata["amount"] == 4.5
    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["account_id"] == "acc1"
    assert unit.metadata["account_name"] == "Checking"
    assert unit.metadata["category"] == ["Food", "Coffee"]
    assert unit.metadata["pending"] == "false"
    assert unit.metadata["source_file"] == "plaid.csv"


def test_plaid_transactions_csv_supports_fallback_ids_filters_directory_and_registry(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("date,name,amount\n2026-05-01,Old,1\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("date,name,amount\n2026-05-03,New,2\n", encoding="utf-8")

    adapter = PlaidTransactionsCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="plaid_transactions_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = adapter.ingest(since=since)
    assert [unit.metadata["name"] for unit in result.units] == ["New"]
    assert result.units[0].source_id.startswith("plaid_transactions_csv:")
    assert adapter.ingest(entity_types=["brokerage_activity"]).units == []
    assert "plaid_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("plaid-transactions-csv"), PlaidTransactionsCsvAdapter)
