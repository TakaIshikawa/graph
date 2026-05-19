from __future__ import annotations

from graph.adapters.citi_credit_card_transactions_csv import CitiCreditCardTransactionsCsvAdapter


def test_citi_credit_card_transactions_csv_ingests_debit_and_credit_rows(tmp_path):
    export = tmp_path / "citi.csv"
    export.write_text(
        "Status,Date,Description,Debit,Credit,Category,Member Name,Reference Number\n"
        "Cleared,04/01/2026,COFFEE SHOP,4.50,,Food & Drink,Jane Example,C100\n"
        "Cleared,04/03/2026,PAYMENT THANK YOU,,25.00,Payment,Jane Example,C101\n"
        "Cleared,,MISSING DATE,1.00,,Other,Jane Example,C102\n"
        "Cleared,04/04/2026,,1.00,,Other,Jane Example,C103\n",
        encoding="utf-8",
    )

    result = CitiCreditCardTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    debit, credit = result.units
    assert debit.source_project == "citi_credit_card_transactions_csv"
    assert debit.source_entity_type == "transaction"
    assert debit.source_id == "citi_credit_card_transactions_csv:C100"
    assert debit.metadata["date"] == "2026-04-01"
    assert debit.metadata["description"] == "COFFEE SHOP"
    assert debit.metadata["category"] == "Food & Drink"
    assert debit.metadata["status"] == "Cleared"
    assert debit.metadata["member_name"] == "Jane Example"
    assert debit.metadata["debit"] == 4.5
    assert debit.metadata["amount"] == -4.5
    assert debit.metadata["currency"] == "USD"
    assert debit.metadata["reference"] == "C100"
    assert debit.metadata["source_file"] == "citi.csv"
    assert debit.tags[:4] == ["finance", "transaction", "citi", "credit-card"]
    assert "COFFEE SHOP" in debit.title
    assert "Food & Drink" in debit.content
    assert "-4.5 USD" in debit.content
    assert credit.metadata["credit"] == 25.0
    assert credit.metadata["amount"] == 25.0


def test_citi_credit_card_transactions_csv_ingests_directory_and_fallback_ids(tmp_path):
    nested = tmp_path / "exports"
    nested.mkdir()
    export = nested / "citi.csv"
    export.write_text(
        "Date,Description,Amount,Category,Member Name\n"
        "2026-04-01,COFFEE SHOP,($4.50),Food & Drink,Jane Example\n",
        encoding="utf-8",
    )

    first = CitiCreditCardTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]
    second = CitiCreditCardTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("citi_credit_card_transactions_csv:")
    assert first.metadata["amount"] == -4.5
    assert first.metadata["source_file"] == "citi.csv"
