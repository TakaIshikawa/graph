from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.xero_bank_transactions_csv import XeroBankTransactionsCsvAdapter
from graph.types.models import SyncState


def test_xero_bank_transactions_csv_ingests_common_headers_and_amounts(tmp_path):
    export = tmp_path / "xero.csv"
    export.write_text(
        "Date,Payee,Description,Reference,Account,Spent,Received,Amount,Tax,Currency,Status,Reconciled\n"
        "2026-04-01,Coffee Shop,Team coffee,R1,Checking,12.50,,,1.00,USD,Reconciled,Yes\n"
        "2026-04-02,Client,Invoice,R2,Checking,,200.00,,0.00,USD,Reconciled,Yes\n",
        encoding="utf-8",
    )

    spent, received = XeroBankTransactionsCsvAdapter(path=str(export)).ingest().units

    assert spent.source_project == "xero_bank_transactions_csv"
    assert spent.source_entity_type == "transaction"
    assert spent.metadata["amount"] == -12.5
    assert received.metadata["amount"] == 200.0
    assert spent.metadata["tax"] == 1.0
    assert "xero" in spent.tags
    assert "Amount: -12.5" in spent.content


def test_xero_bank_transactions_csv_alternate_headers_since_and_minimal_rows(tmp_path):
    export = tmp_path / "xero.csv"
    export.write_text(
        "Transaction Date,Contact,Details,Ref,Bank Account,Debit,Credit,Reconciliation Status\n"
        "2026-04-03,New Payee,New payment,R2,Operating,5.00,,Unreconciled\n"
        "2026-04-01,Old Payee,Old payment,R1,Operating,,10.00,Reconciled\n"
        ",,,,,,,\n"
        ",Only Payee,,,,,,\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="xero_bank_transactions_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 4, 2, tzinfo=timezone.utc))

    units = XeroBankTransactionsCsvAdapter(path=str(export)).ingest(since=since).units

    assert [unit.metadata["payee"] for unit in units] == ["New Payee"]
    assert units[0].metadata["amount"] == -5.0
    all_units = XeroBankTransactionsCsvAdapter(path=str(export)).ingest().units
    assert [unit.metadata["payee"] for unit in all_units] == ["Old Payee", "New Payee", "Only Payee"]
    assert XeroBankTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["note"]).units == []
