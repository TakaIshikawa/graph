from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.schwab_transactions_csv import SchwabTransactionsCsvAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_schwab_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "schwab.csv"
    export.write_text(
        "Date,Action,Symbol,Description,Quantity,Price,Fees & Comm,Amount\n"
        "2026-05-01,Buy,AAPL,APPLE INC,2,150,0,-300\n",
        encoding="utf-8",
    )

    result = SchwabTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.SCHWAB_TRANSACTIONS_CSV
    assert unit.source_entity_type == "brokerage_transaction"
    assert unit.metadata["timestamp"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["action"] == "Buy"
    assert unit.metadata["symbol"] == "AAPL"
    assert unit.metadata["description"] == "APPLE INC"
    assert unit.metadata["quantity"] == 2.0
    assert unit.metadata["price"] == 150.0
    assert unit.metadata["fees"] == 0.0
    assert unit.metadata["amount"] == -300.0
    assert unit.metadata["source_file"] == "schwab.csv"


def test_schwab_transactions_csv_handles_bad_files_filters_ids_and_registry(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "bad.csv").write_bytes(b"\xff\xfe\x00")
    (export_dir / "old.csv").write_text("Date,Action,Amount\n2026-05-01,Old,1\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("Date,Transaction ID,Action,Amount\n2026-05-03,S1,New,2\n", encoding="utf-8")

    adapter = SchwabTransactionsCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="schwab_transactions_csv", source_entity_type="brokerage_transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = adapter.ingest(since=since)
    assert [unit.metadata["action"] for unit in result.units] == ["New"]
    assert result.units[0].source_id == "schwab_transactions_csv:S1"
    assert adapter.ingest(entity_types=["transaction"]).units == []
    assert "schwab_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("schwab-transactions-csv"), SchwabTransactionsCsvAdapter)
