from __future__ import annotations

from graph.adapters.cash_app_activity_csv import CashAppActivityCsvAdapter


def test_cash_app_activity_csv_ingests_payment_and_asset_rows(tmp_path):
    export = tmp_path / "cash-app.csv"
    export.write_text(
        "Date,Transaction Type,Name,Amount,Currency,Status,Notes,Fee,Asset Type,Asset Price,Transaction ID\n"
        "2026-05-01T08:00:00Z,Payment Received,Grace,$25.00,USD,Complete,Dinner,, , ,cash_1\n"
        "05/02/2026,Bitcoin Buy,Cash App Investing,($10.00),USD,Complete,DCA,$0.25,Bitcoin,$65000.00,cash_2\n"
        "\n",
        encoding="utf-8",
    )

    result = CashAppActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    payment, asset = result.units
    assert payment.source_project == "cash_app_activity_csv"
    assert payment.source_entity_type == "transaction"
    assert payment.source_id == "cash_app_activity_csv:cash_1"
    assert payment.metadata["timestamp"] == "2026-05-01T08:00:00+00:00"
    assert payment.metadata["transaction_type"] == "Payment Received"
    assert payment.metadata["name"] == "Grace"
    assert payment.metadata["amount"] == 25.0
    assert payment.metadata["currency"] == "USD"
    assert payment.metadata["status"] == "Complete"
    assert payment.metadata["notes"] == "Dinner"
    assert payment.metadata["source_row"]["Transaction ID"] == "cash_1"
    assert payment.tags[:3] == ["finance", "transaction", "cash-app"]
    assert asset.source_id == "cash_app_activity_csv:cash_2"
    assert asset.metadata["amount"] == -10.0
    assert asset.metadata["fee"] == 0.25
    assert asset.metadata["asset_type"] == "Bitcoin"
    assert asset.metadata["asset_price"] == 65000.0


def test_cash_app_activity_csv_uses_digest_without_transaction_id(tmp_path):
    export = tmp_path / "cash-app.csv"
    export.write_text(
        "Date,Transaction Type,Name,Amount,Currency,Status,Notes,Fee\n"
        "2026-05-03,Payment Sent,Ada,-12.50,USD,Complete,Lunch,\n",
        encoding="utf-8",
    )

    first = CashAppActivityCsvAdapter(path=str(export)).ingest().units[0]
    second = CashAppActivityCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("cash_app_activity_csv:")
    assert first.metadata["amount"] == -12.5
    assert "fee" not in first.metadata
