from __future__ import annotations

from graph.adapters.apple_card_transactions_csv import AppleCardTransactionsCsvAdapter


def test_apple_card_transactions_csv_ingests_purchases_refunds_and_payments(tmp_path):
    export = tmp_path / "apple-card.csv"
    export.write_text(
        "Transaction Date,Clearing Date,Description,Merchant,Category,Type,Amount (USD),Purchased By,Last Four Digits\n"
        "04/01/2026,04/02/2026,Coffee,Cafe,Food & Drink,Purchase,-4.50,Ada,1234\n"
        "04/03/2026,04/04/2026,Returned cable,Electronics Store,Shopping,Refund,19.99,Ada,1234\n"
        "04/05/2026,04/05/2026,Payment,Apple Card,Payment,Payment,100.00,Ada,\n"
        "\n",
        encoding="utf-8",
    )

    result = AppleCardTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 3
    purchase, refund, payment = result.units
    assert purchase.source_project == "apple_card_transactions_csv"
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
    assert purchase.metadata["source_row"]["Amount (USD)"] == "-4.50"
    assert purchase.tags[:3] == ["finance", "transaction", "apple-card"]
    assert refund.metadata["type"] == "Refund"
    assert refund.metadata["amount"] == 19.99
    assert payment.metadata["category"] == "Payment"
    assert payment.metadata["amount"] == 100.0
    assert "last_four_digits" not in payment.metadata


def test_apple_card_transactions_csv_source_ids_are_deterministic(tmp_path):
    export = tmp_path / "apple-card.csv"
    export.write_text(
        "Transaction ID,Transaction Date,Description,Merchant,Category,Type,Amount (USD),Location,Purchased By,Last Four Digits\n"
        "tx_123,2026-04-01,Coffee,Cafe,Food & Drink,Purchase,($4.50),Oakland,Ada,1234\n"
        ",2026-04-02,Tea,Cafe,Food & Drink,Purchase,3.25,Oakland,Ada,1234\n",
        encoding="utf-8",
    )

    first_units = AppleCardTransactionsCsvAdapter(path=str(export)).ingest().units
    second_units = AppleCardTransactionsCsvAdapter(path=str(export)).ingest().units

    assert [unit.source_id for unit in first_units] == [unit.source_id for unit in second_units]
    assert first_units[0].source_id == "apple_card_transactions_csv:tx_123"
    assert first_units[0].metadata["transaction_id"] == "tx_123"
    assert first_units[0].metadata["location"] == "Oakland"
    assert first_units[0].metadata["amount"] == -4.5
    assert first_units[1].source_id.startswith("apple_card_transactions_csv:")
