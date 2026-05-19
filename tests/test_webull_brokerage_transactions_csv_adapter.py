from __future__ import annotations

from graph.adapters.webull_brokerage_transactions_csv import WebullBrokerageTransactionsCsvAdapter


def test_webull_brokerage_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "webull.csv"
    export.write_text(
        "Trade Date,Settlement Date,Symbol,Name,Action,Quantity,Price,Amount,Fees,Type,Transaction ID\n"
        "04/01/2026,04/03/2026,AAPL,APPLE INC,Buy,2,150.25,-301.50,0.00,Trade,W100\n",
        encoding="utf-8",
    )

    result = WebullBrokerageTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "webull_brokerage_transactions_csv"
    assert unit.source_entity_type == "brokerage_transaction"
    assert unit.source_id == "webull_brokerage_transactions_csv:W100"
    assert unit.metadata["trade_date"] == "2026-04-01"
    assert unit.metadata["timestamp"] == "2026-04-01T00:00:00+00:00"
    assert unit.metadata["settlement_date"] == "2026-04-03"
    assert unit.metadata["symbol"] == "AAPL"
    assert unit.metadata["name"] == "APPLE INC"
    assert unit.metadata["action"] == "Buy"
    assert unit.metadata["quantity"] == 2.0
    assert unit.metadata["price"] == 150.25
    assert unit.metadata["amount"] == -301.5
    assert unit.metadata["fees"] == 0.0
    assert unit.metadata["type"] == "Trade"
    assert unit.metadata["transaction_id"] == "W100"
    assert unit.metadata["source_file"] == "webull.csv"
    assert unit.metadata["source_row"]["Transaction ID"] == "W100"
    assert unit.tags[:3] == ["finance", "webull", "brokerage_transaction"]
    assert "Buy AAPL" in unit.title
    assert "APPLE INC" in unit.content


def test_webull_brokerage_transactions_csv_ingests_directories_and_digest_fallback(tmp_path):
    nested = tmp_path / "exports" / "nested"
    nested.mkdir(parents=True)
    export = nested / "webull.csv"
    export.write_text(
        "Trade Date,Settlement Date,Symbol,Name,Action,Quantity,Price,Amount,Fees,Type\n"
        ",,,,,,,,,\n"
        "2026-04-02,2026-04-04,MSFT,MICROSOFT CORP,Sell,1,300,300,0.01,Trade\n",
        encoding="utf-8",
    )

    first = WebullBrokerageTransactionsCsvAdapter(path=str(tmp_path)).ingest().units
    second = WebullBrokerageTransactionsCsvAdapter(path=str(tmp_path)).ingest().units

    assert len(first) == 1
    assert first[0].source_id == second[0].source_id
    assert first[0].source_id.startswith("webull_brokerage_transactions_csv:")
    assert first[0].metadata["amount"] == 300.0
    assert first[0].metadata["fees"] == 0.01
    assert first[0].metadata["source_file"] == "webull.csv"
