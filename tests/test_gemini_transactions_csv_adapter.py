from __future__ import annotations

from graph.adapters.gemini_transactions_csv import GeminiTransactionsCsvAdapter


def test_gemini_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "gemini.csv"
    export.write_text(
        "Date,Type,Symbol,Specification,Liquidity Indicator,Trading Fee,USD Amount,Amount,Fee,Balance,Transaction ID\n"
        "2026-05-01 12:30:00,Buy,BTC,BTCUSD,Auction,1.25,-500.00,0.01,0.00001,0.25,GEM123\n"
        "2026-05-02T08:00:00Z,Deposit,ETH,,,0,250.00,1.5,0,3.0,GEM124\n"
        "\n",
        encoding="utf-8",
    )

    result = GeminiTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    buy, deposit = result.units
    assert buy.source_project == "gemini_transactions_csv"
    assert buy.source_entity_type == "transaction"
    assert buy.source_id == "gemini_transactions_csv:GEM123"
    assert buy.metadata["transaction_id"] == "GEM123"
    assert buy.metadata["timestamp"] == "2026-05-01T12:30:00+00:00"
    assert buy.metadata["transaction_type"] == "Buy"
    assert buy.metadata["symbol"] == "BTC"
    assert buy.metadata["specification"] == "BTCUSD"
    assert buy.metadata["liquidity_indicator"] == "Auction"
    assert buy.metadata["trading_fee"] == 1.25
    assert buy.metadata["usd_amount"] == -500.0
    assert buy.metadata["amount"] == 0.01
    assert buy.metadata["fee"] == 0.00001
    assert buy.metadata["balance"] == 0.25
    assert buy.metadata["source_file"] == "gemini.csv"
    assert buy.metadata["source_row"]["Transaction ID"] == "GEM123"
    assert buy.tags[:4] == ["finance", "crypto", "gemini", "transaction"]
    assert "Buy BTC" in buy.title
    assert "BTCUSD" in buy.content
    assert deposit.source_id == "gemini_transactions_csv:GEM124"
    assert deposit.metadata["amount"] == 1.5


def test_gemini_transactions_csv_ingests_directories_and_ignores_empty_rows(tmp_path):
    nested = tmp_path / "exports" / "nested"
    nested.mkdir(parents=True)
    export = nested / "gemini.csv"
    export.write_text(
        "Date,Type,Symbol,Specification,Liquidity Indicator,Trading Fee,USD Amount,Amount,Fee,Balance\n"
        ",,,,,,,,,\n"
        "2026-05-02,Sell,ETH,ETHUSD,Continuous,0.25,750.00,-0.5,0.001,2.5\n",
        encoding="utf-8",
    )

    first = GeminiTransactionsCsvAdapter(path=str(tmp_path)).ingest().units
    second = GeminiTransactionsCsvAdapter(path=str(tmp_path)).ingest().units

    assert len(first) == 1
    assert first[0].source_id == second[0].source_id
    assert first[0].source_id.startswith("gemini_transactions_csv:")
    assert first[0].metadata["symbol"] == "ETH"
    assert first[0].metadata["source_file"] == "gemini.csv"
