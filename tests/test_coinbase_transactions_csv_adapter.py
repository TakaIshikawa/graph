from __future__ import annotations

from graph.adapters.coinbase_transactions_csv import CoinbaseTransactionsCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject


def test_coinbase_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "coinbase.csv"
    export.write_text(
        "Timestamp,Transaction ID,Transaction Type,Asset,Quantity Transacted,Spot Price at Transaction,Subtotal,Total (inclusive of fees and/or spread),Fees and/or Spread,Notes,Total Currency\n"
        "2026-05-01T08:00:00Z,C1,Buy,BTC,0.01,60000,600,610,10,Recurring buy,USD\n",
        encoding="utf-8",
    )

    result = CoinbaseTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.COINBASE_TRANSACTIONS_CSV
    assert unit.source_id == "coinbase_transactions_csv:C1"
    assert unit.metadata["timestamp"] == "2026-05-01T08:00:00+00:00"
    assert unit.metadata["transaction_type"] == "Buy"
    assert unit.metadata["asset"] == "BTC"
    assert unit.metadata["quantity"] == 0.01
    assert unit.metadata["spot_price"] == 60000.0
    assert unit.metadata["subtotal"] == 600.0
    assert unit.metadata["total"] == 610.0
    assert unit.metadata["fees"] == 10.0
    assert "coinbase" in unit.tags
    assert "crypto" in unit.tags
    assert "BTC" in unit.tags


def test_coinbase_transactions_csv_accepts_common_column_aliases(tmp_path):
    export = tmp_path / "coinbase.csv"
    export.write_text(
        "Date,Type,Currency,Amount,Price,Total,Fee,Description\n"
        "2026-05-02,Reward,ETH,0.5,3000,1500,0,Staking reward\n",
        encoding="utf-8",
    )

    result = CoinbaseTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["transaction_type"] == "Reward"
    assert unit.metadata["asset"] == "ETH"
    assert unit.metadata["quantity"] == 0.5
    assert unit.metadata["spot_price"] == 3000.0
    assert unit.metadata["total"] == 1500.0
    assert unit.metadata["fees"] == 0.0


def test_coinbase_transactions_csv_emits_asset_aggregates_and_edges(tmp_path):
    export = tmp_path / "coinbase.csv"
    export.write_text(
        "Date,Transaction ID,Type,Asset,Quantity,Total,Total Currency\n"
        "2026-05-01,C1,Buy,BTC,0.01,610,USD\n"
        "2026-05-03,C2,Sell,btc,-0.005,-350,USD\n"
        "2026-05-04,C3,Reward,,1,20,USD\n",
        encoding="utf-8",
    )

    result = CoinbaseTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["transaction", "asset"])

    assets = [unit for unit in result.units if unit.source_entity_type == "asset"]
    assert len(assets) == 1
    asset = assets[0]
    assert asset.metadata["asset"] == "BTC"
    assert asset.metadata["transaction_count"] == 2
    assert asset.metadata["total_quantity"] == 0.005
    assert asset.metadata["native_amount_total"] == 260.0
    assert asset.metadata["native_currencies"] == ["USD"]
    assert asset.metadata["first_seen"] == "2026-05-01T00:00:00+00:00"
    assert asset.metadata["last_seen"] == "2026-05-03T00:00:00+00:00"
    assert asset.metadata["transaction_types"] == ["Buy", "Sell"]
    assert {(edge.from_unit_id, edge.to_unit_id, edge.metadata["relation_type"]) for edge in result.edges} == {
        ("coinbase_transactions_csv:C1", asset.source_id, "transaction_asset"),
        ("coinbase_transactions_csv:C2", asset.source_id, "transaction_asset"),
    }

    assert [unit.source_entity_type for unit in CoinbaseTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["asset"]).units] == ["asset"]
    assert CoinbaseTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["transaction"]).edges == []


def test_coinbase_transactions_csv_is_registered():
    assert "coinbase_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("coinbase-transactions-csv"), CoinbaseTransactionsCsvAdapter)
