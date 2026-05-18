from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.ally_bank_transactions_csv import AllyBankTransactionsCsvAdapter
from graph.types.models import SyncState


def test_ally_bank_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "ally.csv"
    export.write_text(
        "Date,Time,Amount,Type,Description,Check Number,Balance,Account\n"
        "2026-05-01,08:15:00,4.50,Debit,Coffee,1001,\"1,245.50\",Interest Checking\n"
        "2026-05-02,09:30:00,\"1,000.00\",Deposit,Payroll,,\"2,245.50\",Interest Checking\n",
        encoding="utf-8",
    )

    result = AllyBankTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    coffee, payroll = result.units
    assert coffee.source_project == "ally_bank_transactions_csv"
    assert coffee.source_entity_type == "transaction"
    assert coffee.metadata["date"] == "2026-05-01"
    assert coffee.metadata["timestamp"] == "2026-05-01T08:15:00+00:00"
    assert coffee.metadata["time"] == "08:15:00"
    assert coffee.metadata["amount"] == -4.5
    assert coffee.metadata["balance"] == 1245.5
    assert coffee.metadata["type"] == "Debit"
    assert coffee.metadata["description"] == "Coffee"
    assert coffee.metadata["check_number"] == "1001"
    assert coffee.metadata["account"] == "Interest Checking"
    assert coffee.metadata["source_file"] == "ally.csv"
    assert coffee.metadata["source_row"]["Balance"] == "1,245.50"
    assert {"ally", "transaction", "Interest Checking", "Debit"}.issubset(set(coffee.tags))
    assert payroll.metadata["amount"] == 1000.0
    assert payroll.metadata["balance"] == 2245.5


def test_ally_bank_transactions_csv_sparse_rows_directory_since_and_entity_filter(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Date,Description,Amount,Type\n2026-05-01,Old coffee,5.00,Debit\n", encoding="utf-8")
    (export_dir / "new.csv").write_text(
        "Date,Amount,Type,Description,Account\n"
        "2026-05-03,($3.25),Withdrawal,ATM cash,Online Savings\n"
        ",,,,,\n",
        encoding="utf-8",
    )

    adapter = AllyBankTransactionsCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="ally_bank_transactions_csv", source_entity_type="transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    first = adapter.ingest(since=since)
    second = adapter.ingest(since=since)

    assert [unit.metadata["description"] for unit in first.units] == ["ATM cash"]
    assert first.units[0].metadata["amount"] == -3.25
    assert first.units[0].metadata["account"] == "Online Savings"
    assert first.units[0].source_id.startswith("ally_bank_transactions_csv:")
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["note"]).units == []


def test_ally_bank_transactions_csv_skips_blank_and_unreadable_files(tmp_path):
    (tmp_path / "blank.csv").write_text("Date,Time,Amount,Type,Description,Check Number,Balance,Account\n,,,,,,,\n", encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    assert AllyBankTransactionsCsvAdapter(path=str(tmp_path)).ingest().units == []
