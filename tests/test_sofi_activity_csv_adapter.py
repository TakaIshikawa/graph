from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.sofi_activity_csv import SofiActivityCsvAdapter
from graph.types.models import SyncState


def test_sofi_activity_csv_ingests_bank_and_brokerage_rows(tmp_path):
    export = tmp_path / "sofi.csv"
    export.write_text(
        "Date,Activity ID,Account,Description,Activity Type,Amount,Status\n"
        "2026-05-01,bank-1,Checking,Payroll,Deposit,2500,Completed\n"
        "2026-05-02,trade-1,Invest,Sofi Technologies,Buy,-120.50,Executed\n",
        encoding="utf-8",
    )
    export.write_text(
        "Date,Activity ID,Account,Description,Activity Type,Amount,Status,Symbol,Security,Quantity,Price,Fees,Currency\n"
        "2026-05-01,bank-1,Checking,Payroll,Deposit,2500,Completed,,,,,,USD\n"
        "2026-05-02,trade-1,Invest,Sofi Technologies,Buy,-120.50,Executed,SOFI,Sofi Technologies Inc,10,12.05,0.10,USD\n",
        encoding="utf-8",
    )

    units = SofiActivityCsvAdapter(path=str(export)).ingest().units

    bank = units[0]
    trade = units[1]
    assert bank.source_id == "sofi_activity_csv:bank-1"
    assert bank.metadata["account"] == "Checking"
    assert bank.metadata["description"] == "Payroll"
    assert bank.metadata["activity_type"] == "Deposit"
    assert bank.metadata["amount"] == 2500
    assert bank.metadata["status"] == "Completed"
    assert trade.metadata["symbol"] == "SOFI"
    assert trade.metadata["security"] == "Sofi Technologies Inc"
    assert trade.metadata["quantity"] == 10
    assert trade.metadata["price"] == 12.05
    assert trade.metadata["fees"] == 0.10
    assert trade.metadata["source_row"]["Activity ID"] == "trade-1"


def test_sofi_activity_csv_directory_filters_and_fallback_ids(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Date,Account,Description,Amount\n2026-05-01,Checking,Old,-1\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("Date,Account,Description,Amount\n2026-05-03,Checking,New,-2\n", encoding="utf-8")

    adapter = SofiActivityCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="sofi_activity_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    assert [unit.metadata["description"] for unit in adapter.ingest(since=since).units] == ["New"]
    assert adapter.ingest(entity_types=["account"]).units == []
    assert [unit.source_id for unit in adapter.ingest().units] == [unit.source_id for unit in adapter.ingest().units]


def test_sofi_activity_csv_is_registered():
    assert "sofi_activity_csv" in list_adapters()
    assert isinstance(get_adapter("sofi-activity-csv"), SofiActivityCsvAdapter)
