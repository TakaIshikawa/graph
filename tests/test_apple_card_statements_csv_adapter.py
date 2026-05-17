from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.apple_card_statements_csv import AppleCardStatementsCsvAdapter
from graph.types.models import SyncState


def test_apple_card_statements_csv_ingests_statement_rows(tmp_path):
    export = tmp_path / "statement.csv"
    export.write_text(
        "Transaction Date,Clearing Date,Description,Merchant,Category,Type,Amount (USD),Purchased By,Last Four Digits\n"
        "04/01/2026,04/02/2026,Coffee,Cafe,Food & Drink,Purchase,-4.50,Ada,1234\n"
        "04/03/2026,04/04/2026,Returned cable,Electronics Store,Shopping,Refund,19.99,Ada,1234\n"
        "04/05/2026,04/05/2026,Payment,Apple Card,Payment,Payment,100.00,Ada,\n"
        "\n",
        encoding="utf-8",
    )

    result = AppleCardStatementsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 3
    purchase, refund, payment = result.units
    assert purchase.source_project == "apple_card_statements_csv"
    assert purchase.source_entity_type == "transaction"
    assert purchase.metadata["transaction_date"] == "2026-04-01"
    assert purchase.metadata["clearing_date"] == "2026-04-02"
    assert purchase.metadata["description"] == "Coffee"
    assert purchase.metadata["merchant"] == "Cafe"
    assert purchase.metadata["category"] == "Food & Drink"
    assert purchase.metadata["type"] == "Purchase"
    assert purchase.metadata["amount"] == -4.5
    assert purchase.metadata["currency"] == "USD"
    assert purchase.metadata["purchased_by"] == "Ada"
    assert purchase.metadata["last_four_digits"] == "1234"
    assert purchase.metadata["source_file"] == "statement.csv"
    assert purchase.metadata["source_row"]["Amount (USD)"] == "-4.50"
    assert purchase.tags[:3] == ["finance", "transaction", "apple-card"]
    assert refund.metadata["type"] == "Refund"
    assert refund.metadata["amount"] == 19.99
    assert payment.metadata["category"] == "Payment"
    assert payment.metadata["amount"] == 100.0
    assert "last_four_digits" not in payment.metadata


def test_apple_card_statements_csv_directory_since_ids_and_entity_filter(tmp_path):
    (tmp_path / "old.csv").write_text(
        "Transaction Date,Description,Amount (USD)\n2026-04-01,Old coffee,-5.00\n",
        encoding="utf-8",
    )
    (tmp_path / "new.csv").write_text(
        "Date,Posted Date,Description,Merchant Name,Category,Transaction Type,Amount USD,User,Card Last Four\n"
        "2026-04-03,2026-04-04,Tea,Cafe,Food,Purchase,($3.25),Ada,1234\n",
        encoding="utf-8",
    )

    adapter = AppleCardStatementsCsvAdapter(path=str(tmp_path))
    since = SyncState(
        source_project="apple_card_statements_csv",
        source_entity_type="transaction",
        last_sync_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )

    first = adapter.ingest(since=since)
    second = adapter.ingest(since=since)

    assert len(first.units) == 1
    assert first.units[0].metadata["merchant"] == "Cafe"
    assert first.units[0].metadata["clearing_date"] == "2026-04-04"
    assert first.units[0].metadata["amount"] == -3.25
    assert first.units[0].metadata["last_four_digits"] == "1234"
    assert first.units[0].source_id.startswith("apple_card_statements_csv:")
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["payment"]).units == []


def test_apple_card_statements_csv_skips_blank_and_unreadable_files(tmp_path):
    (tmp_path / "blank.csv").write_text("Transaction Date,Description,Amount (USD)\n,,\n", encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    assert AppleCardStatementsCsvAdapter(path=str(tmp_path)).ingest().units == []
