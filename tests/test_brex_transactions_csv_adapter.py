from __future__ import annotations

from graph.adapters.brex_transactions_csv import BrexTransactionsCsvAdapter


def test_brex_transactions_csv_ingests_business_spend_rows(tmp_path):
    export = tmp_path / "brex.csv"
    export.write_text(
        "Date,Merchant,Description,Amount,Currency,Status,Category,Card,Employee,Memo,Receipt URL,Transaction ID\n"
        "04/01/2026,Cloud Vendor,Monthly hosting,-45.00,USD,Approved,Software,Virtual 1234,Ada Lovelace,April infra,https://receipt.example/b100,B100\n"
        "04/02/2026,,Wire transfer,2500.00,USD,Posted,Revenue,Cash account,Grace Hopper,,https://receipt.example/b101,B101\n"
        "\n",
        encoding="utf-8",
    )

    result = BrexTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    cloud, wire = result.units
    assert cloud.source_project == "brex_transactions_csv"
    assert cloud.source_entity_type == "transaction"
    assert cloud.source_id == "brex_transactions_csv:B100"
    assert cloud.metadata["transaction_id"] == "B100"
    assert cloud.metadata["date"] == "2026-04-01"
    assert cloud.metadata["timestamp"] == "2026-04-01T00:00:00+00:00"
    assert cloud.metadata["merchant"] == "Cloud Vendor"
    assert cloud.metadata["description"] == "Monthly hosting"
    assert cloud.metadata["amount"] == -45.0
    assert cloud.metadata["currency"] == "USD"
    assert cloud.metadata["status"] == "Approved"
    assert cloud.metadata["category"] == "Software"
    assert cloud.metadata["card"] == "Virtual 1234"
    assert cloud.metadata["employee"] == "Ada Lovelace"
    assert cloud.metadata["memo"] == "April infra"
    assert cloud.metadata["receipt_url"] == "https://receipt.example/b100"
    assert cloud.metadata["source_file"] == "brex.csv"
    assert cloud.metadata["source_row"]["Transaction ID"] == "B100"
    assert cloud.tags[:3] == ["finance", "transaction", "brex"]
    assert "Cloud Vendor" in cloud.title
    assert "-45 USD" in cloud.title
    assert "April infra" in cloud.content
    assert wire.title.startswith("Wire transfer")
    assert wire.metadata["amount"] == 2500.0


def test_brex_transactions_csv_uses_deterministic_digest_fallback(tmp_path):
    nested = tmp_path / "exports"
    nested.mkdir()
    export = nested / "brex.csv"
    export.write_text(
        "Date,Merchant,Description,Amount,Currency,Status,Category,Card,Employee,Memo,Receipt URL\n"
        "2026-04-03,Airline,Customer visit,($199.99),USD,Pending,Travel,Physical 9876,Ada Lovelace,Sales trip,\n",
        encoding="utf-8",
    )

    first = BrexTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]
    second = BrexTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("brex_transactions_csv:")
    assert first.metadata["amount"] == -199.99
    assert first.metadata["merchant"] == "Airline"
    assert first.metadata["source_file"] == "brex.csv"
