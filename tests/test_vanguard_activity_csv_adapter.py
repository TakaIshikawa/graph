from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.vanguard_activity_csv import VanguardActivityCsvAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_vanguard_activity_csv_ingests_activity_rows(tmp_path):
    export = tmp_path / "vanguard.csv"
    export.write_text(
        "Trade Date,Settlement Date,Transaction Type,Investment Name,Symbol,Shares,Share Price,Principal Amount,Net Amount\n"
        "2026-05-01,2026-05-03,Buy,Vanguard Total Stock,VTSAX,2,150,-300,-300\n",
        encoding="utf-8",
    )

    result = VanguardActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.VANGUARD_ACTIVITY_CSV
    assert unit.source_entity_type == "brokerage_activity"
    assert unit.metadata["trade_date"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["settlement_date"] == "2026-05-03T00:00:00+00:00"
    assert unit.metadata["transaction_type"] == "Buy"
    assert unit.metadata["investment_name"] == "Vanguard Total Stock"
    assert unit.metadata["symbol"] == "VTSAX"
    assert unit.metadata["shares"] == 2.0
    assert unit.metadata["share_price"] == 150.0
    assert unit.metadata["principal_amount"] == -300.0
    assert unit.metadata["net_amount"] == -300.0
    assert unit.metadata["source_file"] == "vanguard.csv"


def test_vanguard_activity_csv_supports_filters_directory_ids_and_registry(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Trade Date,Transaction Type,Net Amount\n2026-05-01,Old,1\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("Trade Date,Activity ID,Transaction Type,Net Amount\n2026-05-03,V1,New,2\n", encoding="utf-8")

    adapter = VanguardActivityCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="vanguard_activity_csv", source_entity_type="brokerage_activity", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = adapter.ingest(since=since)
    assert [unit.metadata["transaction_type"] for unit in result.units] == ["New"]
    assert result.units[0].source_id == "vanguard_activity_csv:V1"
    assert adapter.ingest(entity_types=["transaction"]).units == []
    assert "vanguard_activity_csv" in list_adapters()
    assert isinstance(get_adapter("vanguard-activity-csv"), VanguardActivityCsvAdapter)
