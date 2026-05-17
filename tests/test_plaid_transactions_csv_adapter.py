from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.plaid_transactions_csv import PlaidTransactionsCsvAdapter
from graph.types.models import SyncState


def test_plaid_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "plaid.csv"
    export.write_text(
        "transaction_id,date,authorized_date,name,merchant_name,amount,iso_currency_code,account_id,account_name,category,pending,payment_channel,location_city,location_region,location_country,website\n"
        "tx1,2026-05-01,2026-04-30,Coffee purchase,Cafe,-4.50,USD,acc1,Checking,\"Food, Coffee\",false,in store,San Francisco,CA,US,https://cafe.example\n",
        encoding="utf-8",
    )

    result = PlaidTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "plaid_transactions_csv"
    assert unit.source_entity_type == "transaction"
    assert unit.source_id == "plaid_transactions_csv:tx1"
    assert unit.metadata["date"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["authorized_date"] == "2026-04-30T00:00:00+00:00"
    assert unit.metadata["name"] == "Coffee purchase"
    assert unit.metadata["merchant_name"] == "Cafe"
    assert unit.metadata["amount"] == -4.5
    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["account_id"] == "acc1"
    assert unit.metadata["account_name"] == "Checking"
    assert unit.metadata["category"] == ["Food", "Coffee"]
    assert unit.metadata["pending"] is False
    assert unit.metadata["payment_channel"] == "in store"
    assert unit.metadata["location_city"] == "San Francisco"
    assert unit.metadata["location_region"] == "CA"
    assert unit.metadata["location_country"] == "US"
    assert unit.metadata["website"] == "https://cafe.example"
    assert unit.metadata["source_file"] == "plaid.csv"
    assert {"plaid", "transaction", "Checking", "Cafe", "Food", "Coffee"}.issubset(set(unit.tags))


def test_plaid_transactions_csv_supports_aliases_fallback_ids_filters_and_directory(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Transaction Date,Description,Amount\n2026-05-01,Old,1\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("Posted Date,Merchant,Amount,Account,Category,Pending,Channel,City,State,Country,URL\n2026-05-03,New merchant,2.25,Savings,Transfer,true,online,Tokyo,Tokyo,JP,https://example.test\n", encoding="utf-8")

    adapter = PlaidTransactionsCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="plaid_transactions_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = adapter.ingest(since=since)
    assert [unit.metadata["merchant_name"] for unit in result.units] == ["New merchant"]
    assert result.units[0].metadata["merchant_name"] == "New merchant"
    assert result.units[0].metadata["pending"] is True
    assert result.units[0].metadata["location_country"] == "JP"
    assert result.units[0].source_id.startswith("plaid_transactions_csv:")
    assert [unit.source_id for unit in result.units] == [unit.source_id for unit in adapter.ingest(since=since).units]
    assert adapter.ingest(entity_types=["brokerage_activity"]).units == []


def test_plaid_transactions_csv_skips_empty_and_unreadable_files(tmp_path):
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    assert PlaidTransactionsCsvAdapter(path=str(tmp_path)).ingest().units == []
