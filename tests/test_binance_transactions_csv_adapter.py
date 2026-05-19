from __future__ import annotations

from graph.adapters.binance_transactions_csv import BinanceTransactionsCsvAdapter


def test_binance_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "binance.csv"
    export.write_text(
        "UTC_Time,Account,Operation,Coin,Change,Remark,Transaction ID,Order ID\n"
        "2026-05-01 08:00:00,Spot,Deposit,BTC,0.5,On-chain deposit,BN123,ORD1\n"
        "2026-05-01T09:00:00Z,Funding,Commission,BNB,-0.01,Trading fee,,ORD2\n"
        "\n",
        encoding="utf-8",
    )

    result = BinanceTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    deposit, fee = result.units
    assert deposit.source_project == "binance_transactions_csv"
    assert deposit.source_entity_type == "transaction"
    assert deposit.source_id == "binance_transactions_csv:BN123"
    assert deposit.metadata["transaction_id"] == "BN123"
    assert deposit.metadata["order_id"] == "ORD1"
    assert deposit.metadata["utc_timestamp"] == "2026-05-01T08:00:00+00:00"
    assert deposit.metadata["account"] == "Spot"
    assert deposit.metadata["operation"] == "Deposit"
    assert deposit.metadata["coin"] == "BTC"
    assert deposit.metadata["change"] == 0.5
    assert deposit.metadata["remark"] == "On-chain deposit"
    assert deposit.metadata["source_file"] == "binance.csv"
    assert deposit.metadata["source_row"]["Transaction ID"] == "BN123"
    assert deposit.tags[:4] == ["finance", "crypto", "binance", "transaction"]
    assert "Deposit BTC" in deposit.title
    assert "On-chain deposit" in deposit.content
    assert fee.source_id == "binance_transactions_csv:ORD2"
    assert fee.metadata["change"] == -0.01


def test_binance_transactions_csv_uses_deterministic_digest_fallback(tmp_path):
    nested = tmp_path / "exports"
    nested.mkdir()
    export = nested / "binance.csv"
    export.write_text(
        "UTC_Time,Account,Operation,Coin,Change,Remark\n"
        "2026-05-02,Spot,Withdraw,ETH,-1.25,External wallet\n",
        encoding="utf-8",
    )

    first = BinanceTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]
    second = BinanceTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("binance_transactions_csv:")
    assert first.metadata["coin"] == "ETH"
    assert first.metadata["change"] == -1.25
    assert first.metadata["source_file"] == "binance.csv"
