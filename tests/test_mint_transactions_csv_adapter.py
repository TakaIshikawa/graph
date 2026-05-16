from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.mint_transactions_csv import MintTransactionsCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_mint_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "mint.csv"
    export.write_text(
        "Date,Description,Original Description,Amount,Transaction Type,Category,Account Name,Labels,Notes\n"
        "2026-05-01,Coffee,CAFE,4.50,debit,Food,Checking,\"coffee,work\",Flat white\n"
        "2026-05-02,Paycheck,EMPLOYER,2500,credit,Income,Checking,,\n",
        encoding="utf-8",
    )

    result = MintTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    unit = result.units[0]
    assert unit.source_project == SourceProject.MINT_TRANSACTIONS_CSV
    assert unit.source_entity_type == "transaction"
    assert unit.metadata["timestamp"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["description"] == "Coffee"
    assert unit.metadata["original_description"] == "CAFE"
    assert unit.metadata["amount"] == -4.5
    assert unit.metadata["transaction_type"] == "debit"
    assert unit.metadata["category"] == "Food"
    assert unit.metadata["account"] == "Checking"
    assert unit.metadata["labels"] == ["coffee", "work"]
    assert unit.metadata["notes"] == "Flat white"
    assert unit.metadata["source_file"] == "mint.csv"
    assert result.units[1].metadata["amount"] == 2500.0


def test_mint_transactions_csv_supports_directory_since_entity_and_registry(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Date,Description,Amount,Transaction Type\n2026-05-01,Old,1,debit\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("Date,Description,Amount,Transaction Type\n2026-05-03,New,2,credit\n", encoding="utf-8")

    adapter = MintTransactionsCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="mint_transactions_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    assert [unit.metadata["description"] for unit in adapter.ingest(since=since).units] == ["New"]
    assert adapter.ingest(entity_types=["note"]).units == []
    assert "mint_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("mint-transactions-csv"), MintTransactionsCsvAdapter)
