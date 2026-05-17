from __future__ import annotations

from graph.adapters.chase_transactions_csv import ChaseTransactionsCsvAdapter


def test_chase_transactions_csv_ingests_credit_and_debit_rows(tmp_path):
    export = tmp_path / "chase.csv"
    export.write_text(
        "Transaction Date,Post Date,Description,Category,Type,Amount,Memo,Check or Slip #\n"
        "04/01/2026,04/02/2026,COFFEE SHOP,Food & Drink,Sale,-4.50,Latte,\n"
        "04/03/2026,04/04/2026,PAYMENT THANK YOU,Payment,Payment,100.00,,12345\n"
        "\n",
        encoding="utf-8",
    )

    result = ChaseTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    purchase, payment = result.units
    assert purchase.source_project == "chase_transactions_csv"
    assert purchase.source_entity_type == "transaction"
    assert purchase.metadata["transaction_date"] == "2026-04-01"
    assert purchase.metadata["post_date"] == "2026-04-02"
    assert purchase.metadata["description"] == "COFFEE SHOP"
    assert purchase.metadata["category"] == "Food & Drink"
    assert purchase.metadata["type"] == "Sale"
    assert purchase.metadata["amount"] == -4.5
    assert purchase.metadata["currency"] == "USD"
    assert purchase.metadata["memo"] == "Latte"
    assert purchase.metadata["source_row"]["Amount"] == "-4.50"
    assert purchase.tags[:3] == ["finance", "transaction", "chase"]
    assert payment.metadata["amount"] == 100.0
    assert payment.metadata["check_or_slip_number"] == "12345"


def test_chase_transactions_csv_source_ids_are_deterministic(tmp_path):
    export = tmp_path / "chase.csv"
    export.write_text(
        "Transaction Date,Post Date,Description,Category,Type,Amount,Memo,Check or Slip #\n"
        "2026-04-01,2026-04-02,COFFEE SHOP,Food & Drink,Sale,($4.50),Latte,\n",
        encoding="utf-8",
    )

    first = ChaseTransactionsCsvAdapter(path=str(export)).ingest().units[0]
    second = ChaseTransactionsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("chase_transactions_csv:")
    assert first.metadata["amount"] == -4.5
