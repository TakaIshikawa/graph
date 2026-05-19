from __future__ import annotations

from graph.adapters.discover_credit_card_transactions_csv import DiscoverCreditCardTransactionsCsvAdapter


def test_discover_credit_card_transactions_csv_ingests_charge_and_payment_rows(tmp_path):
    export = tmp_path / "discover.csv"
    export.write_text(
        "Trans. Date,Post Date,Description,Amount,Category,Transaction Type,Reference Number\n"
        "04/01/2026,04/02/2026,COFFEE SHOP,4.50,Restaurants,Sale,D100\n"
        "04/03/2026,04/04/2026,INTERNET PAYMENT,25.00,Payment,Payment,D101\n",
        encoding="utf-8",
    )

    result = DiscoverCreditCardTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    charge, payment = result.units
    assert charge.source_project == "discover_credit_card_transactions_csv"
    assert charge.source_entity_type == "transaction"
    assert charge.source_id == "discover_credit_card_transactions_csv:D100"
    assert charge.metadata["transaction_date"] == "2026-04-01"
    assert charge.metadata["post_date"] == "2026-04-02"
    assert charge.metadata["description"] == "COFFEE SHOP"
    assert charge.metadata["category"] == "Restaurants"
    assert charge.metadata["transaction_type"] == "Sale"
    assert charge.metadata["amount"] == -4.5
    assert charge.metadata["currency"] == "USD"
    assert charge.metadata["reference"] == "D100"
    assert charge.metadata["source_file"] == "discover.csv"
    assert charge.tags[:4] == ["finance", "transaction", "discover", "credit-card"]
    assert "COFFEE SHOP" in charge.title
    assert "Restaurants" in charge.content
    assert "-4.5 USD" in charge.content
    assert payment.metadata["transaction_type"] == "Payment"
    assert payment.metadata["amount"] == 25.0


def test_discover_credit_card_transactions_csv_ingests_directory_and_fallback_ids(tmp_path):
    nested = tmp_path / "exports"
    nested.mkdir()
    export = nested / "discover.csv"
    export.write_text(
        "Trans. Date,Post Date,Description,Amount,Category,Transaction Type\n"
        "2026-04-01,2026-04-02,COFFEE SHOP,($4.50),Restaurants,Sale\n",
        encoding="utf-8",
    )

    first = DiscoverCreditCardTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]
    second = DiscoverCreditCardTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("discover_credit_card_transactions_csv:")
    assert first.metadata["amount"] == -4.5
    assert first.metadata["source_file"] == "discover.csv"
