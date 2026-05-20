from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.interactive_brokers_activity_csv import InteractiveBrokersActivityCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.models import SyncState


def test_interactive_brokers_activity_csv_ingests_supported_sections(tmp_path):
    export = tmp_path / "ibkr.csv"
    export.write_text(
        "Section,Type,Date,Account,Asset Class,Symbol,Description,Quantity,Price,Proceeds,Currency,Commission,Transaction ID\n"
        "Statement Header,Header,2026-05-01,U123,,,,,,,,,\n"
        "Trades,Trade,2026-05-01,U123,Stocks,AAPL,BUY AAPL,2,180,-360,USD,-1,ib-1\n"
        "Dividends,Dividend,2026-05-02,U123,Stocks,AAPL,AAPL Dividend,,,4.20,USD,,ib-2\n"
        "Fees,Fee,2026-05-03,U123,,,Monthly fee,,,-10,USD,,ib-3\n"
        "Deposits & Withdrawals,Deposit,2026-05-04,U123,,,ACH Deposit,,,1000,USD,,ib-4\n",
        encoding="utf-8",
    )

    units = InteractiveBrokersActivityCsvAdapter(path=str(export)).ingest().units

    assert [unit.metadata["type"] for unit in units] == ["Trade", "Dividend", "Fee", "Deposit"]
    trade = units[0]
    assert trade.source_id == "interactive_brokers_activity_csv:ib-1"
    assert trade.metadata["account"] == "U123"
    assert trade.metadata["asset_class"] == "Stocks"
    assert trade.metadata["symbol"] == "AAPL"
    assert trade.metadata["quantity"] == 2
    assert trade.metadata["price"] == 180
    assert trade.metadata["amount"] == -360
    assert trade.metadata["currency"] == "USD"
    assert trade.metadata["commission"] == -1
    assert trade.metadata["section"] == "Trades"
    assert trade.metadata["source_row"]["Transaction ID"] == "ib-1"


def test_interactive_brokers_activity_csv_sparse_directory_filters_and_fallback_ids(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Section,Type,Date,Account,Description,Amount\nInterest,Interest,2026-05-01,U123,Credit interest,1.25\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("Section,Type,Date,Account,Description,Amount\nTax,Tax,2026-05-03,U123,Withholding tax,-0.50\n", encoding="utf-8")
    (export_dir / "summary.csv").write_text("Section,Type,Date,Description,Amount\nStatement Summary,Summary,2026-05-04,Totals,999\n", encoding="utf-8")

    adapter = InteractiveBrokersActivityCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="interactive_brokers_activity_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    assert [unit.metadata["description"] for unit in adapter.ingest(since=since).units] == ["Withholding tax"]
    assert adapter.ingest(entity_types=["note"]).units == []
    assert [unit.source_id for unit in adapter.ingest().units] == [unit.source_id for unit in adapter.ingest().units]


def test_interactive_brokers_activity_csv_is_registered():
    assert "interactive_brokers_activity_csv" in list_adapters()
    assert isinstance(get_adapter("interactive-brokers-activity-csv"), InteractiveBrokersActivityCsvAdapter)
