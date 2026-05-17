from __future__ import annotations

from graph.adapters.ramp_transactions_csv import RampTransactionsCsvAdapter


def test_ramp_transactions_csv_ingests_corporate_card_rows(tmp_path):
    export = tmp_path / "ramp.csv"
    export.write_text(
        "Transaction ID,Date,Merchant,Amount,Currency,User,Department,Memo,Category,Receipt,Card Last Four,State/Status\n"
        "R123,04/01/2026,Cloud Vendor,-45.00,USD,Ada Lovelace,Engineering,Monthly hosting,Software,https://receipt.example/r123,1234,Approved\n"
        "R124,04/03/2026,Airline,-199.99,USD,Grace Hopper,Sales,Customer visit,Travel,,9876,Pending\n"
        "\n",
        encoding="utf-8",
    )

    result = RampTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    cloud, airline = result.units
    assert cloud.source_project == "ramp_transactions_csv"
    assert cloud.source_id == "ramp_transactions_csv:R123"
    assert cloud.source_entity_type == "transaction"
    assert cloud.metadata["transaction_id"] == "R123"
    assert cloud.metadata["date"] == "2026-04-01"
    assert cloud.metadata["merchant"] == "Cloud Vendor"
    assert cloud.metadata["amount"] == -45.0
    assert cloud.metadata["currency"] == "USD"
    assert cloud.metadata["user"] == "Ada Lovelace"
    assert cloud.metadata["department"] == "Engineering"
    assert cloud.metadata["memo"] == "Monthly hosting"
    assert cloud.metadata["category"] == "Software"
    assert cloud.metadata["receipt"] == "https://receipt.example/r123"
    assert cloud.metadata["card_last_four"] == "1234"
    assert cloud.metadata["status"] == "Approved"
    assert cloud.metadata["source_row"]["Transaction ID"] == "R123"
    assert cloud.tags[:3] == ["finance", "transaction", "ramp"]
    assert airline.metadata["status"] == "Pending"
    assert airline.metadata["card_last_four"] == "9876"


def test_ramp_transactions_csv_source_ids_are_deterministic_without_transaction_id(tmp_path):
    export = tmp_path / "ramp.csv"
    export.write_text(
        "Date,Merchant,Amount,Currency,User,Department,Card Last Four,State/Status\n"
        "2026-04-01,Cloud Vendor,($45.00),USD,Ada Lovelace,Engineering,1234,Approved\n",
        encoding="utf-8",
    )

    first = RampTransactionsCsvAdapter(path=str(export)).ingest().units[0]
    second = RampTransactionsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("ramp_transactions_csv:")
    assert first.metadata["amount"] == -45.0
