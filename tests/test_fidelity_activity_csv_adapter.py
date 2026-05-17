from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.fidelity_activity_csv import FidelityActivityCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_fidelity_activity_csv_ingests_activity_rows(tmp_path):
    export = tmp_path / "fidelity.csv"
    export.write_text(
        "Run Date,Action,Symbol,Description,Quantity,Price ($),Amount ($),Settlement Date\n"
        "2026-05-01,BUY,AAPL,APPLE INC,2,150,-300,2026-05-03\n",
        encoding="utf-8",
    )

    result = FidelityActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.FIDELITY_ACTIVITY_CSV
    assert unit.source_entity_type == "brokerage_activity"
    assert unit.metadata["run_date"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["settlement_date"] == "2026-05-03T00:00:00+00:00"
    assert unit.metadata["action"] == "BUY"
    assert unit.metadata["symbol"] == "AAPL"
    assert unit.metadata["description"] == "APPLE INC"
    assert unit.metadata["quantity"] == 2.0
    assert unit.metadata["price"] == 150.0
    assert unit.metadata["amount"] == -300.0
    assert unit.metadata["source_file"] == "fidelity.csv"


def test_fidelity_activity_csv_supports_filters_directory_ids_and_registry(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Run Date,Action,Amount\n2026-05-01,OLD,1\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("Run Date,Activity ID,Action,Amount\n2026-05-03,F1,NEW,2\n", encoding="utf-8")

    adapter = FidelityActivityCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="fidelity_activity_csv", source_entity_type="brokerage_activity", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = adapter.ingest(since=since)
    assert [unit.metadata["action"] for unit in result.units] == ["NEW"]
    assert result.units[0].source_id == "fidelity_activity_csv:F1"
    assert adapter.ingest(entity_types=["transaction"]).units == []
    assert "fidelity_activity_csv" in list_adapters()
    assert isinstance(get_adapter("fidelity-activity-csv"), FidelityActivityCsvAdapter)
