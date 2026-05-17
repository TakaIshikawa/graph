from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.robinhood_activity_csv import RobinhoodActivityCsvAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_robinhood_activity_csv_ingests_activity_rows(tmp_path):
    export = tmp_path / "robinhood.csv"
    export.write_text(
        "Activity Date,Process Date,Instrument,Description,Trans Code,Quantity,Price,Amount\n"
        "2026-05-01,2026-05-02,AAPL,Apple buy,Buy,2,150,-300\n",
        encoding="utf-8",
    )

    result = RobinhoodActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.ROBINHOOD_ACTIVITY_CSV
    assert unit.source_entity_type == "brokerage_activity"
    assert unit.metadata["activity_date"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["process_date"] == "2026-05-02T00:00:00+00:00"
    assert unit.metadata["instrument"] == "AAPL"
    assert unit.metadata["description"] == "Apple buy"
    assert unit.metadata["transaction_code"] == "Buy"
    assert unit.metadata["quantity"] == 2.0
    assert unit.metadata["price"] == 150.0
    assert unit.metadata["amount"] == -300.0
    assert unit.metadata["source_file"] == "robinhood.csv"


def test_robinhood_activity_csv_supports_filters_directory_ids_and_registry(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Activity Date,Instrument,Amount\n2026-05-01,OLD,1\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("Activity Date,Activity ID,Instrument,Amount\n2026-05-03,RH1,NEW,2\n", encoding="utf-8")

    adapter = RobinhoodActivityCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="robinhood_activity_csv", source_entity_type="brokerage_activity", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = adapter.ingest(since=since)
    assert [unit.metadata["instrument"] for unit in result.units] == ["NEW"]
    assert result.units[0].source_id == "robinhood_activity_csv:RH1"
    assert adapter.ingest(entity_types=["transaction"]).units == []
    assert "robinhood_activity_csv" in list_adapters()
    assert isinstance(get_adapter("robinhood-activity-csv"), RobinhoodActivityCsvAdapter)
