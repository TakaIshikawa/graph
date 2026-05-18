from __future__ import annotations

from graph.adapters.chase_credit_card_transactions_csv import ChaseCreditCardTransactionsCsvAdapter


def test_chase_credit_card_transactions_csv_ingests_file_rows(tmp_path):
    export = tmp_path / "chase.csv"
    export.write_text(
        "Transaction Date,Post Date,Description,Category,Type,Amount,Memo,Account,Card,Transaction ID\n"
        "04/01/2026,04/02/2026,COFFEE SHOP,Food & Drink,Sale,4.50,Latte,Freedom,1234,T100\n"
        "04/03/2026,04/04/2026,PAYMENT THANK YOU,Payment,Payment,25.00,,Freedom,1234,T101\n",
        encoding="utf-8",
    )

    result = ChaseCreditCardTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    purchase, payment = result.units
    assert purchase.source_project == "chase_credit_card_transactions_csv"
    assert purchase.source_entity_type == "transaction"
    assert purchase.source_id == "chase_credit_card_transactions_csv:T100"
    assert purchase.metadata["transaction_date"] == "2026-04-01"
    assert purchase.metadata["post_date"] == "2026-04-02"
    assert purchase.metadata["description"] == "COFFEE SHOP"
    assert purchase.metadata["category"] == "Food & Drink"
    assert purchase.metadata["type"] == "Sale"
    assert purchase.metadata["amount"] == -4.5
    assert purchase.metadata["memo"] == "Latte"
    assert purchase.metadata["account"] == "Freedom"
    assert purchase.metadata["card"] == "1234"
    assert purchase.metadata["transaction_id"] == "T100"
    assert purchase.metadata["source_file"] == "chase.csv"
    assert purchase.tags[:4] == ["finance", "transaction", "chase", "credit-card"]
    assert "COFFEE SHOP" in purchase.title
    assert payment.metadata["amount"] == 25.0


def test_chase_credit_card_transactions_csv_ingests_directory_and_fallback_ids(tmp_path):
    nested = tmp_path / "exports"
    nested.mkdir()
    export = nested / "chase.csv"
    export.write_text(
        "Transaction Date,Post Date,Description,Category,Type,Amount\n"
        "2026-04-01,2026-04-02,COFFEE SHOP,Food & Drink,Sale,($4.50)\n",
        encoding="utf-8",
    )

    first = ChaseCreditCardTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]
    second = ChaseCreditCardTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("chase_credit_card_transactions_csv:")
    assert first.source_id != "chase_credit_card_transactions_csv:"
    assert first.metadata["amount"] == -4.5
    assert first.metadata["source_file"] == "chase.csv"
