from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.acorns_activity_csv import AcornsActivityCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.models import SyncState


def test_acorns_activity_csv_ingests_activity_rows(tmp_path):
    export = tmp_path / "acorns.csv"
    export.write_text(
        "Date,Account,Activity Type,Description,Merchant,Category,Round-Up Amount,Investment Amount,Total Amount,Status,Transaction ID\n"
        "04/01/2026,Invest,Round-Up,Card purchase round-up,Coffee Shop,Food & Drink,0.45,0.45,4.50,Completed,A100\n"
        "04/02/2026,Later,Recurring Investment,Weekly deposit,,Investing,,25.00,25.00,Pending,A101\n"
        "\n",
        encoding="utf-8",
    )

    result = AcornsActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    round_up, recurring = result.units
    assert round_up.source_project == "acorns_activity_csv"
    assert round_up.source_entity_type == "transaction"
    assert round_up.source_id == "acorns_activity_csv:A100"
    assert round_up.metadata["date"] == "2026-04-01"
    assert round_up.metadata["timestamp"] == "2026-04-01T00:00:00+00:00"
    assert round_up.metadata["account"] == "Invest"
    assert round_up.metadata["activity_type"] == "Round-Up"
    assert round_up.metadata["description"] == "Card purchase round-up"
    assert round_up.metadata["merchant"] == "Coffee Shop"
    assert round_up.metadata["category"] == "Food & Drink"
    assert round_up.metadata["round_up_amount"] == 0.45
    assert round_up.metadata["investment_amount"] == 0.45
    assert round_up.metadata["total_amount"] == 4.5
    assert round_up.metadata["currency"] == "USD"
    assert round_up.metadata["status"] == "Completed"
    assert round_up.metadata["transaction_id"] == "A100"
    assert round_up.metadata["source_file"] == "acorns.csv"
    assert round_up.metadata["source_row"]["Merchant"] == "Coffee Shop"
    assert round_up.tags[:3] == ["finance", "transaction", "acorns"]
    assert "Round-Up - Coffee Shop" in round_up.title
    assert "Round-Up Amount: 0.45" in round_up.content
    assert recurring.metadata["investment_amount"] == 25.0
    assert recurring.metadata["total_amount"] == 25.0
    assert recurring.metadata["account"] == "Later"


def test_acorns_activity_csv_directory_filters_skips_and_digest_fallback(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text(
        "Date,Account,Activity Type,Description,Total Amount\n"
        "2026-04-01,Invest,Round-Up,Old purchase,1.00\n",
        encoding="utf-8",
    )
    (export_dir / "new.csv").write_text(
        "Date,Account,Activity Type,Description,Total Amount\n"
        "2026-04-03,Invest,Round-Up,New purchase,2.00\n",
        encoding="utf-8",
    )
    (export_dir / "blank.csv").write_text("", encoding="utf-8")
    (export_dir / "unreadable.csv").write_bytes(b"\xff\xfe\x00")

    adapter = AcornsActivityCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="acorns_activity_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 4, 2, tzinfo=timezone.utc))

    result = adapter.ingest(since=since)
    assert [unit.metadata["description"] for unit in result.units] == ["New purchase"]
    first = result.units[0]
    second = AcornsActivityCsvAdapter(path=str(export_dir)).ingest(since=since).units[0]
    assert first.source_id == second.source_id
    assert first.source_id.startswith("acorns_activity_csv:")
    assert first.metadata["source_file"] == "new.csv"
    assert adapter.ingest(entity_types=["note"]).units == []
    assert "acorns_activity_csv" in list_adapters()
    assert isinstance(get_adapter("acorns-activity-csv"), AcornsActivityCsvAdapter)
