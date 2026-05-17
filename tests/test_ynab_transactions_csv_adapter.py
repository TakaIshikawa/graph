from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.ynab_transactions_csv import YnabTransactionsCsvAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_ynab_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "ynab.csv"
    export.write_text(
        "Account,Date,Payee,Category Group/Category,Memo,Outflow,Inflow,Cleared\n"
        "Checking,2026-05-01,Grocer,Everyday: Groceries,Weekly shop,42.25,,Cleared\n"
        "Checking,2026-05-02,Employer,Income: Paycheck,, ,2500,Uncleared\n",
        encoding="utf-8",
    )

    result = YnabTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    grocery = result.units[0]
    assert grocery.source_project == SourceProject.YNAB_TRANSACTIONS_CSV
    assert grocery.source_entity_type == "transaction"
    assert grocery.source_id.startswith("ynab_transactions_csv:")
    assert grocery.metadata["timestamp"] == "2026-05-01T00:00:00+00:00"
    assert grocery.metadata["account"] == "Checking"
    assert grocery.metadata["payee"] == "Grocer"
    assert grocery.metadata["category"] == "Everyday: Groceries"
    assert grocery.metadata["memo"] == "Weekly shop"
    assert grocery.metadata["outflow"] == 42.25
    assert grocery.metadata["amount"] == -42.25
    assert grocery.metadata["cleared"] == "Cleared"
    assert grocery.metadata["source_file"] == "ynab.csv"
    assert "ynab" in grocery.tags
    assert "Everyday: Groceries" in grocery.tags
    assert result.units[1].metadata["amount"] == 2500.0


def test_ynab_transactions_csv_supports_directory_since_and_entity_filters(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "older.csv").write_text(
        "Account,Date,Payee,Category,Outflow,Inflow,Cleared\nChecking,2026-05-01,Old,Food,1,,Cleared\n",
        encoding="utf-8",
    )
    (export_dir / "newer.csv").write_text(
        "Account,Date,Payee,Category,Outflow,Inflow,Cleared\nChecking,2026-05-03,New,Food,,2,Cleared\n",
        encoding="utf-8",
    )

    adapter = YnabTransactionsCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="ynab_transactions_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    assert [unit.metadata["payee"] for unit in adapter.ingest(since=since).units] == ["New"]
    assert adapter.ingest(entity_types=["note"]).units == []


def test_ynab_transactions_csv_uses_transaction_id_and_is_registered(tmp_path):
    export = tmp_path / "ynab.csv"
    export.write_text(
        "Date,Transaction ID,Payee,Amount\n2026-05-01,abc123,Transfer,12.50\n",
        encoding="utf-8",
    )

    result = YnabTransactionsCsvAdapter(path=str(export)).ingest()

    assert result.units[0].source_id == "ynab_transactions_csv:abc123"
    assert "ynab_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("ynab-transactions-csv"), YnabTransactionsCsvAdapter)
