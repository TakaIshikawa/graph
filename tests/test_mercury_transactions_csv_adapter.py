from __future__ import annotations

from graph.adapters.mercury_transactions_csv import MercuryTransactionsCsvAdapter


def test_mercury_transactions_csv_ingests_banking_rows(tmp_path):
    export = tmp_path / "mercury.csv"
    export.write_text(
        "Date,Description,Amount,Balance,Bank Description,Category,Note,Status,Counterparty Name,Counterparty Account,Reference ID\n"
        "04/01/2026,ACH debit,-45.00,955.00,ACH WEB PAYMENT,Software,Monthly subscription,Posted,Vendor Inc,****1234,M123\n"
        "04/03/2026,Wire credit,1000.00,1955.00,INCOMING WIRE,Income,Invoice paid,Posted,Client LLC,****9876,M124\n"
        "\n",
        encoding="utf-8",
    )

    result = MercuryTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    debit, credit = result.units
    assert debit.source_project == "mercury_transactions_csv"
    assert debit.source_id == "mercury_transactions_csv:M123"
    assert debit.source_entity_type == "transaction"
    assert debit.metadata["date"] == "2026-04-01"
    assert debit.metadata["description"] == "ACH debit"
    assert debit.metadata["amount"] == -45.0
    assert debit.metadata["currency"] == "USD"
    assert debit.metadata["balance"] == 955.0
    assert debit.metadata["bank_description"] == "ACH WEB PAYMENT"
    assert debit.metadata["category"] == "Software"
    assert debit.metadata["note"] == "Monthly subscription"
    assert debit.metadata["status"] == "Posted"
    assert debit.metadata["counterparty_name"] == "Vendor Inc"
    assert debit.metadata["counterparty_account"] == "****1234"
    assert debit.metadata["reference_id"] == "M123"
    assert debit.metadata["source_row"]["Counterparty Name"] == "Vendor Inc"
    assert debit.tags[:3] == ["finance", "transaction", "mercury"]
    assert credit.metadata["amount"] == 1000.0
    assert credit.metadata["counterparty_name"] == "Client LLC"


def test_mercury_transactions_csv_source_ids_are_deterministic_without_reference(tmp_path):
    export = tmp_path / "mercury.csv"
    export.write_text(
        "Date,Description,Amount,Balance,Counterparty Name,Counterparty Account\n"
        "2026-04-01,ACH debit,($45.00),955.00,Vendor Inc,****1234\n",
        encoding="utf-8",
    )

    first = MercuryTransactionsCsvAdapter(path=str(export)).ingest().units[0]
    second = MercuryTransactionsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("mercury_transactions_csv:")
    assert first.metadata["amount"] == -45.0
