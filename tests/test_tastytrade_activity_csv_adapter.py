from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.tastytrade_activity_csv import TastytradeActivityCsvAdapter
from graph.types.models import SyncState


def test_tastytrade_activity_csv_ingests_activity_rows(tmp_path):
    export = tmp_path / "tastytrade.csv"
    export.write_text(
        "Date/Time,Account,Action,Symbol,Underlying,Description,Quantity,Price,Amount,Commission,Fees,Currency,Transaction ID,Order ID\n"
        "2026-04-01 09:30:00,Tasty Margin,Buy,AAPL,AAPL,Buy 2 AAPL,2,150.25,-300.50,1.00,0.14,USD,T100,O100\n"
        "2026-04-02 10:15:00,Tasty Margin,Dividend,,AAPL,AAPL dividend,,,12.34,0,0,USD,T101,\n"
        "\n",
        encoding="utf-8",
    )

    result = TastytradeActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    buy, dividend = result.units
    assert buy.source_project == "tastytrade_activity_csv"
    assert buy.source_entity_type == "transaction"
    assert buy.source_id == "tastytrade_activity_csv:T100"
    assert buy.metadata["timestamp"] == "2026-04-01T09:30:00+00:00"
    assert buy.metadata["date"] == "2026-04-01"
    assert buy.metadata["account"] == "Tasty Margin"
    assert buy.metadata["action"] == "Buy"
    assert buy.metadata["symbol"] == "AAPL"
    assert buy.metadata["underlying"] == "AAPL"
    assert buy.metadata["description"] == "Buy 2 AAPL"
    assert buy.metadata["quantity"] == 2.0
    assert buy.metadata["price"] == 150.25
    assert buy.metadata["amount"] == -300.5
    assert buy.metadata["commission"] == 1.0
    assert buy.metadata["fees"] == 0.14
    assert buy.metadata["currency"] == "USD"
    assert buy.metadata["transaction_id"] == "T100"
    assert buy.metadata["order_id"] == "O100"
    assert buy.metadata["source_file"] == "tastytrade.csv"
    assert buy.metadata["source_row"]["Order ID"] == "O100"
    assert buy.tags[:3] == ["finance", "transaction", "tastytrade"]
    assert "Buy AAPL" in buy.title
    assert "-300.5 USD" in buy.title
    assert "Commission: 1.0" in buy.content
    assert dividend.metadata["amount"] == 12.34
    assert dividend.metadata["underlying"] == "AAPL"


def test_tastytrade_activity_csv_directory_filters_skips_and_digest_fallback(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text(
        "Date,Time,Account,Type,Symbol,Amount\n"
        "2026-04-01,09:00:00,Margin,Trade,OLD,-1.00\n",
        encoding="utf-8",
    )
    (export_dir / "new.csv").write_text(
        "Date,Time,Account,Type,Symbol,Amount\n"
        "2026-04-03,09:00:00,Margin,Trade,NEW,-2.00\n",
        encoding="utf-8",
    )
    (export_dir / "blank.csv").write_text("", encoding="utf-8")
    (export_dir / "unreadable.csv").write_bytes(b"\xff\xfe\x00")

    adapter = TastytradeActivityCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="tastytrade_activity_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 4, 2, tzinfo=timezone.utc))

    result = adapter.ingest(since=since)
    assert [unit.metadata["symbol"] for unit in result.units] == ["NEW"]
    first = result.units[0]
    second = TastytradeActivityCsvAdapter(path=str(export_dir)).ingest(since=since).units[0]
    assert first.source_id == second.source_id
    assert first.source_id.startswith("tastytrade_activity_csv:")
    assert first.metadata["timestamp"] == "2026-04-03T09:00:00+00:00"
    assert first.metadata["source_file"] == "new.csv"
    assert adapter.ingest(entity_types=["brokerage_transaction"]).units == []
    assert "tastytrade_activity_csv" in list_adapters()
    assert isinstance(get_adapter("tastytrade-activity-csv"), TastytradeActivityCsvAdapter)


def test_tastytrade_activity_csv_uses_order_id_when_transaction_id_is_missing(tmp_path):
    export = tmp_path / "orders.csv"
    export.write_text(
        "Date,Account,Action,Symbol,Amount,Order ID\n"
        "2026-04-04,Margin,Sell,MSFT,200.00,O200\n",
        encoding="utf-8",
    )

    unit = TastytradeActivityCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id == "tastytrade_activity_csv:order:O200"
    assert unit.metadata["order_id"] == "O200"
