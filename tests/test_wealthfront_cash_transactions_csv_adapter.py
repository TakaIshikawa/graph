from __future__ import annotations

from graph.adapters.wealthfront_cash_transactions_csv import WealthfrontCashTransactionsCsvAdapter


def test_wealthfront_cash_transactions_csv_ingests_rows(tmp_path):
    export = tmp_path / "wealthfront-cash.csv"
    export.write_text(
        "Date,Description,Category,Amount,Balance,Account,Institution,Transaction ID\n"
        "04/01/2026,Interest Payment,Interest,5.25,1005.25,Cash Account,Wealthfront,WFC1\n"
        "04/03/2026,ACH Transfer,Transfer,-100.00,905.25,Cash Account,Wealthfront,WFC2\n"
        "\n",
        encoding="utf-8",
    )

    result = WealthfrontCashTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    interest, transfer = result.units
    assert interest.source_project == "wealthfront_cash_transactions_csv"
    assert interest.source_entity_type == "transaction"
    assert interest.source_id == "wealthfront_cash_transactions_csv:WFC1"
    assert interest.metadata["date"] == "2026-04-01"
    assert interest.metadata["timestamp"] == "2026-04-01T00:00:00+00:00"
    assert interest.metadata["description"] == "Interest Payment"
    assert interest.metadata["category"] == "Interest"
    assert interest.metadata["amount"] == 5.25
    assert interest.metadata["currency"] == "USD"
    assert interest.metadata["balance"] == 1005.25
    assert interest.metadata["account"] == "Cash Account"
    assert interest.metadata["institution"] == "Wealthfront"
    assert interest.metadata["transaction_id"] == "WFC1"
    assert interest.metadata["source_file"] == "wealthfront-cash.csv"
    assert interest.metadata["source_row"]["Description"] == "Interest Payment"
    assert interest.tags[:4] == ["finance", "transaction", "wealthfront", "cash"]
    assert "Interest Payment" in interest.title
    assert "1005.25" in interest.content
    assert transfer.metadata["amount"] == -100.0
    assert transfer.metadata["balance"] == 905.25


def test_wealthfront_cash_transactions_csv_uses_digest_fallback_for_missing_transaction_ids(tmp_path):
    nested = tmp_path / "exports"
    nested.mkdir()
    export = nested / "wealthfront.csv"
    export.write_text(
        "Date,Description,Category,Amount,Balance,Account,Institution\n"
        "2026-04-01,Debit Card Purchase,Shopping,($12.34),892.91,Cash Account,Wealthfront\n",
        encoding="utf-8",
    )

    first = WealthfrontCashTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]
    second = WealthfrontCashTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("wealthfront_cash_transactions_csv:")
    assert first.metadata["amount"] == -12.34
    assert first.metadata["balance"] == 892.91
    assert first.metadata["source_file"] == "wealthfront.csv"
