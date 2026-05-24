from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.quickbooks_transactions_csv import QuickBooksTransactionsCsvAdapter
from graph.types.models import SyncState


def test_quickbooks_transactions_csv_ingests_common_headers_and_amounts(tmp_path):
    export = tmp_path / "quickbooks.csv"
    export.write_text(
        "Date,Transaction Type,Num,Name,Memo,Account,Split,Debit,Credit,Amount,Balance,Class,Location\n"
        "2026-04-01,Expense,1001,Coffee Shop,Team coffee,Checking,Meals,12.50,,,100.00,Ops,NY\n"
        "2026-04-02,Deposit,1002,Client,Invoice,Checking,Sales,,200.00,,300.00,Revenue,SF\n",
        encoding="utf-8",
    )

    expense, deposit = QuickBooksTransactionsCsvAdapter(path=str(export)).ingest().units

    assert expense.source_project == "quickbooks_transactions_csv"
    assert expense.source_entity_type == "transaction"
    assert expense.metadata["amount"] == -12.5
    assert deposit.metadata["amount"] == 200.0
    assert expense.metadata["class"] == "Ops"
    assert "quickbooks" in expense.tags
    assert "Amount: -12.5" in expense.content


def test_quickbooks_transactions_csv_alternate_headers_since_and_minimal_rows(tmp_path):
    export = tmp_path / "quickbooks.csv"
    export.write_text(
        "Date,Type,Number,Vendor,Description,Account,Amount\n"
        "2026-04-03,Bill,2,New Vendor,New bill,AP,-25.00\n"
        "2026-04-01,Bill,1,Old Vendor,Old bill,AP,-10.00\n"
        ",,,,,,\n"
        ",,,Name Only,,,\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="quickbooks_transactions_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 4, 2, tzinfo=timezone.utc))

    units = QuickBooksTransactionsCsvAdapter(path=str(export)).ingest(since=since).units

    assert [unit.metadata["name"] for unit in units] == ["New Vendor"]
    all_units = QuickBooksTransactionsCsvAdapter(path=str(export)).ingest().units
    assert [unit.metadata["name"] for unit in all_units] == ["Old Vendor", "New Vendor", "Name Only"]
    assert QuickBooksTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["note"]).units == []
