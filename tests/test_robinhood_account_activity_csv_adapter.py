from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.robinhood_account_activity_csv import RobinhoodAccountActivityCsvAdapter
from graph.types.models import SyncState


def test_robinhood_account_activity_csv_ingests_buys_sells_and_dividends(tmp_path):
    export = tmp_path / "robinhood.csv"
    export.write_text(
        "Activity Date,Process Date,Symbol,Description,Activity Type,Quantity,Price,Amount,Fees,Currency,Account,Transaction ID\n"
        "2026-05-01,2026-05-02,AAPL,Apple buy,Buy,2,150,-300,0.02,USD,Individual,RH1\n"
        "2026-05-03,2026-05-04,MSFT,Microsoft dividend,Dividend,,,12.50,,USD,Individual,RH2\n",
        encoding="utf-8",
    )

    result = RobinhoodAccountActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    buy = result.units[0]
    assert buy.source_project == "robinhood_account_activity_csv"
    assert buy.source_id == "robinhood_account_activity_csv:RH1"
    assert buy.source_entity_type == "account_activity"
    assert buy.metadata["activity_date"] == "2026-05-01T00:00:00+00:00"
    assert buy.metadata["process_date"] == "2026-05-02T00:00:00+00:00"
    assert buy.metadata["symbol"] == "AAPL"
    assert buy.metadata["quantity"] == 2.0
    assert buy.metadata["price"] == 150.0
    assert buy.metadata["amount"] == -300.0
    assert buy.metadata["fees"] == 0.02
    assert buy.metadata["currency"] == "USD"
    assert {"robinhood", "account_activity", "Buy", "AAPL"}.issubset(set(buy.tags))
    assert "Amount: -300.0 USD" in buy.content
    assert result.units[1].metadata["activity_type"] == "Dividend"
    assert result.units[1].metadata["amount"] == 12.5


def test_robinhood_account_activity_csv_aliases_filters_and_deterministic_fallback(tmp_path):
    (tmp_path / "old.csv").write_text("Trade Date,Ticker,Net Amount\n2026-05-01,OLD,1\n", encoding="utf-8")
    (tmp_path / "new.csv").write_text(
        "Date,Settle Date,Instrument,Name,Transaction Type,Shares,Share Price,Total,Fee,Account Number\n"
        "2026-05-03,2026-05-05,TSLA,Sell shares,Sell,1,700,700,0.01,Acct-9\n",
        encoding="utf-8",
    )

    adapter = RobinhoodAccountActivityCsvAdapter(path=str(tmp_path))
    sync = SyncState(source_project="robinhood_account_activity_csv", source_entity_type="account_activity", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.metadata["symbol"] for unit in first.units] == ["TSLA"]
    assert first.units[0].metadata["activity_type"] == "Sell"
    assert first.units[0].metadata["fees"] == 0.01
    assert first.units[0].source_id.startswith("robinhood_account_activity_csv:")
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["transaction"]).units == []


def test_robinhood_account_activity_csv_sparse_rows(tmp_path):
    export = tmp_path / "sparse.csv"
    export.write_text("Activity Date,Symbol,Amount\n, ,\n2026-05-06,HOOD,5.25\n", encoding="utf-8")

    result = RobinhoodAccountActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["symbol"] == "HOOD"
    assert result.units[0].metadata["amount"] == 5.25
