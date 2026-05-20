from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.m1_finance_activity_csv import M1FinanceActivityCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.models import SyncState


def test_m1_finance_activity_csv_ingests_activity_types(tmp_path):
    export = tmp_path / "m1.csv"
    export.write_text(
        "Date,Activity ID,Account,Symbol,Security Name,Action,Quantity,Price,Amount,Cash Balance,Notes\n"
        "2026-05-01,m1-1,Taxable,VTI,Vanguard Total Stock Market ETF,Buy,2,250,-500,100,Auto-invest\n"
        "2026-05-02,m1-2,Taxable,,Cash,Deposit,,,1000,1100,ACH deposit\n"
        "2026-05-03,m1-3,Taxable,VTI,Vanguard Total Stock Market ETF,Dividend,,,4.25,1104.25,Dividend payment\n",
        encoding="utf-8",
    )

    units = M1FinanceActivityCsvAdapter(path=str(export)).ingest().units

    assert [unit.metadata["action"] for unit in units] == ["Buy", "Deposit", "Dividend"]
    buy = units[0]
    assert buy.source_id == "m1_finance_activity_csv:m1-1"
    assert buy.metadata["symbol"] == "VTI"
    assert buy.metadata["security"] == "Vanguard Total Stock Market ETF"
    assert buy.metadata["quantity"] == 2
    assert buy.metadata["price"] == 250
    assert buy.metadata["amount"] == -500
    assert buy.metadata["account"] == "Taxable"
    assert buy.metadata["cash_balance"] == 100
    assert buy.metadata["source_row"]["Activity ID"] == "m1-1"


def test_m1_finance_activity_csv_directory_malformed_filters_and_fallback_ids(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Date,Account,Action,Amount\n2026-05-01,Taxable,Sell,200\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("Date,Account,Action,Amount\n2026-05-03,Taxable,Deposit,100\n", encoding="utf-8")
    (export_dir / "bad.csv").write_bytes(b"\xff\xfe")

    adapter = M1FinanceActivityCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="m1_finance_activity_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    assert [unit.metadata["action"] for unit in adapter.ingest(since=since).units] == ["Deposit"]
    assert adapter.ingest(entity_types=["note"]).units == []
    assert [unit.source_id for unit in adapter.ingest().units] == [unit.source_id for unit in adapter.ingest().units]


def test_m1_finance_activity_csv_is_registered():
    assert "m1_finance_activity_csv" in list_adapters()
    assert isinstance(get_adapter("m1-finance-activity-csv"), M1FinanceActivityCsvAdapter)
