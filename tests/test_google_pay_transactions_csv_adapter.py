from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.google_pay_transactions_csv import GooglePayTransactionsCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_google_pay_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "google-pay.csv"
    export.write_text(
        "Transaction Date,Transaction ID,Merchant,Description,Amount,Currency,Status,Payment Method,Category,Notes\n"
        "2026-05-01 10:30:00,GP-1,Market Cafe,Coffee,$4.75,USD,Completed,Visa 1234,Food,Morning stop\n",
        encoding="utf-8",
    )

    result = GooglePayTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "google_pay_transactions_csv"
    assert unit.source_entity_type == "transaction"
    assert unit.source_id == "google_pay_transactions_csv:GP-1"
    assert unit.title == "Market Cafe (4.75 USD)"
    assert unit.content_type == ContentType.METADATA
    assert unit.metadata["timestamp"] == "2026-05-01T10:30:00+00:00"
    assert unit.metadata["transaction_id"] == "GP-1"
    assert unit.metadata["merchant"] == "Market Cafe"
    assert unit.metadata["description"] == "Coffee"
    assert unit.metadata["amount"] == 4.75
    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["status"] == "Completed"
    assert unit.metadata["payment_method"] == "Visa 1234"
    assert unit.metadata["category"] == "Food"
    assert unit.metadata["notes"] == "Morning stop"
    assert unit.metadata["source_file"] == "google-pay.csv"
    assert "Merchant: Market Cafe" in unit.content
    assert "Amount: 4.75 USD" in unit.content
    assert unit.tags == ["google_pay", "transaction", "Completed", "Food", "USD"]


def test_google_pay_transactions_csv_accepts_directory_and_deterministic_fallback_ids(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "one.csv").write_text(
        "Date,Merchant,Amount,Currency\n"
        '"May 1, 2026",Transit,-2.50,USD\n',
        encoding="utf-8",
    )
    (export_dir / "two.csv").write_text(
        "Date,Merchant,Amount,Currency\n"
        '"May 2, 2026",Retail,12.00,USD\n',
        encoding="utf-8",
    )
    (export_dir / "ignored.txt").write_text("Date,Merchant\n2026-05-03,Ignored\n", encoding="utf-8")

    first = GooglePayTransactionsCsvAdapter(path=str(export_dir)).ingest()
    second = GooglePayTransactionsCsvAdapter(path=str(export_dir)).ingest()

    assert [unit.metadata["merchant"] for unit in first.units] == ["Transit", "Retail"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert all(unit.source_id.startswith("google_pay_transactions_csv:") for unit in first.units)


def test_google_pay_transactions_csv_supports_since_and_entity_filters(tmp_path):
    export = tmp_path / "google-pay.csv"
    export.write_text(
        "Date,Transaction ID,Merchant,Amount\n"
        "2026-05-01,OLD,Old Shop,1.00\n"
        "2026-05-03,NEW,New Shop,2.00\n",
        encoding="utf-8",
    )

    filtered = GooglePayTransactionsCsvAdapter(path=str(export)).ingest(
        since=SyncState(
            source_project="google_pay_transactions_csv",
            source_entity_type="transaction",
            last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
    )
    wrong_entity = GooglePayTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["brokerage_activity"])

    assert [unit.source_id for unit in filtered.units] == ["google_pay_transactions_csv:NEW"]
    assert wrong_entity.units == []
    assert wrong_entity.edges == []


def test_google_pay_transactions_csv_skips_empty_rows(tmp_path):
    export = tmp_path / "google-pay.csv"
    export.write_text(
        "Date,Merchant,Amount\n"
        ",,\n",
        encoding="utf-8",
    )

    assert GooglePayTransactionsCsvAdapter(path=str(export)).ingest().units == []
