from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.tiller_transactions_csv import TillerTransactionsCsvAdapter
from graph.types.models import SyncState


def test_tiller_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "tiller.csv"
    export.write_text(
        "Date,Description,Category,Amount,Account,Account #,Institution,Month,Week,Transaction ID,Check Number,Full Description\n"
        "2026-05-01,Coffee,Food,-4.50,Checking,1234,Local Bank,2026-05,2026-W18,tx-1,101,Cafe purchase downtown\n"
        "2026-05-02,Paycheck,Income,\"1,250.00\",Checking,1234,Local Bank,2026-05,2026-W18,tx-2,,Payroll deposit\n",
        encoding="utf-8",
    )

    result = TillerTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    coffee, paycheck = result.units
    assert coffee.source_project == "tiller_transactions_csv"
    assert coffee.source_entity_type == "transaction"
    assert coffee.source_id == "tiller_transactions_csv:tx-1"
    assert coffee.metadata["transaction_id"] == "tx-1"
    assert coffee.metadata["date"] == "2026-05-01"
    assert coffee.metadata["timestamp"] == "2026-05-01T00:00:00+00:00"
    assert coffee.metadata["description"] == "Coffee"
    assert coffee.metadata["full_description"] == "Cafe purchase downtown"
    assert coffee.metadata["category"] == "Food"
    assert coffee.metadata["amount"] == -4.5
    assert coffee.metadata["account"] == "Checking"
    assert coffee.metadata["account_number"] == "1234"
    assert coffee.metadata["institution"] == "Local Bank"
    assert coffee.metadata["month"] == "2026-05"
    assert coffee.metadata["week"] == "2026-W18"
    assert coffee.metadata["check_number"] == "101"
    assert coffee.metadata["source_file"] == "tiller.csv"
    assert coffee.metadata["source_row"]["Account #"] == "1234"
    assert {"tiller", "transaction", "Checking", "Local Bank", "Food"}.issubset(set(coffee.tags))
    assert paycheck.metadata["amount"] == 1250.0


def test_tiller_transactions_csv_sparse_rows_directory_since_and_entity_filter(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Date,Description,Amount\n2026-05-01,Old coffee,-5.00\n", encoding="utf-8")
    (export_dir / "new.csv").write_text(
        "Date,Description,Amount,Account,Institution,Category\n"
        "2026-05-03,Tea,($3.25),Savings,Credit Union,Food\n"
        ",,,,,\n",
        encoding="utf-8",
    )

    adapter = TillerTransactionsCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="tiller_transactions_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    first = adapter.ingest(since=since)
    second = adapter.ingest(since=since)

    assert [unit.metadata["description"] for unit in first.units] == ["Tea"]
    assert first.units[0].metadata["amount"] == -3.25
    assert first.units[0].metadata["account"] == "Savings"
    assert first.units[0].metadata["institution"] == "Credit Union"
    assert first.units[0].source_id.startswith("tiller_transactions_csv:")
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["note"]).units == []


def test_tiller_transactions_csv_skips_blank_and_unreadable_files(tmp_path):
    (tmp_path / "blank.csv").write_text("Date,Description,Amount\n,,\n", encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    assert TillerTransactionsCsvAdapter(path=str(tmp_path)).ingest().units == []
