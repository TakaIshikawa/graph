from __future__ import annotations

from graph.adapters.schwab_brokerage_transactions_csv import SchwabBrokerageTransactionsCsvAdapter


def test_schwab_brokerage_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "schwab.csv"
    export.write_text(
        "Date,Action,Symbol,Description,Quantity,Price,Fees & Comm,Amount,Account,Transaction ID\n"
        "04/01/2026,Buy,AAPL,APPLE INC,2,150.25,1.00,-301.50,Brokerage,S100\n",
        encoding="utf-8",
    )

    result = SchwabBrokerageTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "schwab_brokerage_transactions_csv"
    assert unit.source_entity_type == "brokerage_transaction"
    assert unit.source_id == "schwab_brokerage_transactions_csv:S100"
    assert unit.metadata["timestamp"] == "2026-04-01T00:00:00+00:00"
    assert unit.metadata["date"] == "2026-04-01"
    assert unit.metadata["action"] == "Buy"
    assert unit.metadata["symbol"] == "AAPL"
    assert unit.metadata["description"] == "APPLE INC"
    assert unit.metadata["quantity"] == 2.0
    assert unit.metadata["price"] == 150.25
    assert unit.metadata["fees_commission"] == 1.0
    assert unit.metadata["amount"] == -301.5
    assert unit.metadata["account"] == "Brokerage"
    assert unit.metadata["transaction_id"] == "S100"
    assert unit.metadata["source_file"] == "schwab.csv"
    assert unit.tags[:3] == ["finance", "schwab", "brokerage_transaction"]
    assert "Buy AAPL" in unit.title
    assert "-301.5" in unit.title
    assert "APPLE INC" in unit.content


def test_schwab_brokerage_transactions_csv_ingests_directory_trees_and_digest_fallback(tmp_path):
    nested = tmp_path / "exports" / "nested"
    nested.mkdir(parents=True)
    export = nested / "schwab.csv"
    export.write_text(
        "Date,Action,Symbol,Description,Quantity,Price,Fees & Comm,Amount,Account\n"
        "2026-04-01,Sell,MSFT,MICROSOFT CORP,1,300,0,300,Brokerage\n",
        encoding="utf-8",
    )

    first = SchwabBrokerageTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]
    second = SchwabBrokerageTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("schwab_brokerage_transactions_csv:")
    assert first.metadata["amount"] == 300.0
    assert first.metadata["account"] == "Brokerage"
    assert first.metadata["source_file"] == "schwab.csv"
