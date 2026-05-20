from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.personal_capital_transactions_csv import PersonalCapitalTransactionsCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.models import SyncState


def test_personal_capital_transactions_csv_ingests_representative_row(tmp_path):
    export = tmp_path / "pc.csv"
    export.write_text(
        "Date,Transaction ID,Account,Description,Original Description,Category,Amount,Tags,Notes\n"
        '2026-05-01,pc-1,Cash,Corner Market,CORNER MKT 123,Groceries,-42.50,"food,home",Weekly shop\n',
        encoding="utf-8",
    )

    unit = PersonalCapitalTransactionsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "personal_capital_transactions_csv"
    assert unit.source_id == "personal_capital_transactions_csv:pc-1"
    assert unit.source_entity_type == "transaction"
    assert unit.metadata["timestamp"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["account"] == "Cash"
    assert unit.metadata["merchant"] == "Corner Market"
    assert unit.metadata["description"] == "Corner Market"
    assert unit.metadata["original_description"] == "CORNER MKT 123"
    assert unit.metadata["category"] == "Groceries"
    assert unit.metadata["amount"] == -42.50
    assert unit.metadata["tags"] == ["food", "home"]
    assert unit.metadata["notes"] == "Weekly shop"
    assert unit.metadata["source_row"]["Transaction ID"] == "pc-1"


def test_personal_capital_transactions_csv_directory_filters_and_fallback_ids(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Date,Account,Merchant,Amount\n2026-05-01,Cash,Old,-1\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("date,account name,payee,transaction amount\n2026-05-03,Cash,New,-2\n", encoding="utf-8")
    (export_dir / "empty.csv").write_text("", encoding="utf-8")
    (export_dir / "bad.csv").write_bytes(b"\xff\xfe\x00")

    adapter = PersonalCapitalTransactionsCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="personal_capital_transactions_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    assert [unit.metadata["description"] for unit in adapter.ingest(since=since).units] == ["New"]
    assert adapter.ingest(entity_types=["note"]).units == []
    assert [unit.source_id for unit in adapter.ingest().units] == [unit.source_id for unit in adapter.ingest().units]


def test_personal_capital_transactions_csv_is_registered():
    assert "personal_capital_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("personal-capital-transactions-csv"), PersonalCapitalTransactionsCsvAdapter)
