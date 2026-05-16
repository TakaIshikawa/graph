from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.wise_activity_csv import WiseActivityCsvAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_wise_activity_csv_ingests_transfer_rows(tmp_path):
    export = tmp_path / "wise.csv"
    export.write_text(
        "Date,Updated,Transfer ID,Type,Status,Reference,Source Amount,Source Currency,Target Amount,Target Currency,Fee,Exchange Rate,Recipient,Account\n"
        "2026-05-01T08:00:00Z,2026-05-01T08:05:00Z,W1,Transfer,Completed,Invoice 42,100,GBP,124.50,USD,0.75,1.245,Ada,Business\n",
        encoding="utf-8",
    )

    result = WiseActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.WISE_ACTIVITY_CSV
    assert unit.source_id == "wise_activity_csv:W1"
    assert unit.metadata["timestamp"] == "2026-05-01T08:00:00+00:00"
    assert unit.metadata["updated_at"] == "2026-05-01T08:05:00+00:00"
    assert unit.metadata["type"] == "Transfer"
    assert unit.metadata["status"] == "Completed"
    assert unit.metadata["reference"] == "Invoice 42"
    assert unit.metadata["source_amount"] == 100.0
    assert unit.metadata["source_currency"] == "GBP"
    assert unit.metadata["target_amount"] == 124.5
    assert unit.metadata["target_currency"] == "USD"
    assert unit.metadata["fee"] == 0.75
    assert unit.metadata["exchange_rate"] == 1.245
    assert unit.metadata["recipient"] == "Ada"


def test_wise_activity_csv_filters_since_and_skips_blank_rows(tmp_path):
    export = tmp_path / "wise.csv"
    export.write_text(
        "Date,Updated,Transfer ID,Type,Source Amount,Currency\n"
        ",,,,,\n"
        "2026-05-01,2026-05-01,W1,Transfer,10,GBP\n"
        "2026-05-02,2026-05-03,W2,Payment,20,GBP\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="wise_activity_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = WiseActivityCsvAdapter(path=str(export)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == ["wise_activity_csv:W2"]


def test_wise_activity_csv_is_registered():
    assert "wise_activity_csv" in list_adapters()
    assert isinstance(get_adapter("wise-activity-csv"), WiseActivityCsvAdapter)
