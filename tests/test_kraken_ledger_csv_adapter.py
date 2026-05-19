from __future__ import annotations

from graph.adapters.kraken_ledger_csv import KrakenLedgerCsvAdapter


def test_kraken_ledger_csv_ingests_ledger_rows(tmp_path):
    export = tmp_path / "kraken-ledgers.csv"
    export.write_text(
        "txid,refid,time,type,subtype,aclass,asset,amount,fee,balance\n"
        "TX1,REF1,2026-05-01 08:00:00,trade,,currency,XXBT,-0.0100000000,0.0000200000,1.2345000000\n"
        ",REF2,2026-05-01T09:00:00Z,deposit,staking,currency,ZUSD,100.50,0,250.75\n"
        "\n",
        encoding="utf-8",
    )

    result = KrakenLedgerCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    trade, deposit = result.units
    assert trade.source_project == "kraken_ledger_csv"
    assert trade.source_entity_type == "ledger"
    assert trade.source_id == "kraken_ledger_csv:TX1"
    assert trade.metadata["txid"] == "TX1"
    assert trade.metadata["refid"] == "REF1"
    assert trade.metadata["timestamp"] == "2026-05-01T08:00:00+00:00"
    assert trade.metadata["type"] == "trade"
    assert trade.metadata["aclass"] == "currency"
    assert trade.metadata["asset"] == "XXBT"
    assert trade.metadata["amount"] == -0.01
    assert trade.metadata["fee"] == 0.00002
    assert trade.metadata["balance"] == 1.2345
    assert trade.metadata["source_file"] == "kraken-ledgers.csv"
    assert trade.metadata["source_row"]["txid"] == "TX1"
    assert trade.tags[:4] == ["finance", "crypto", "kraken", "ledger"]
    assert "trade" in trade.title
    assert "XXBT" in trade.content
    assert deposit.source_id == "kraken_ledger_csv:REF2"
    assert deposit.metadata["subtype"] == "staking"
    assert deposit.metadata["amount"] == 100.5


def test_kraken_ledger_csv_ingests_directories_and_uses_digest_fallback(tmp_path):
    nested = tmp_path / "exports"
    nested.mkdir()
    malformed = tmp_path / "bad.csv"
    malformed.write_bytes(b"\xff\xfe\x00")
    export = nested / "ledger.csv"
    export.write_text(
        "time,type,subtype,aclass,asset,amount,fee,balance\n"
        "\n"
        "2026-05-02,withdrawal,,currency,ETH,-0.25,0.001,3.75\n",
        encoding="utf-8",
    )

    first = KrakenLedgerCsvAdapter(path=str(tmp_path)).ingest().units[0]
    second = KrakenLedgerCsvAdapter(path=str(tmp_path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("kraken_ledger_csv:")
    assert first.metadata["amount"] == -0.25
    assert first.metadata["fee"] == 0.001
    assert first.metadata["balance"] == 3.75
    assert first.metadata["source_file"] == "ledger.csv"
