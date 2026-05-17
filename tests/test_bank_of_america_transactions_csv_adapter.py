from __future__ import annotations

from graph.adapters.bank_of_america_transactions_csv import BankOfAmericaTransactionsCsvAdapter


def test_bank_of_america_transactions_csv_ingests_rows_with_balances(tmp_path):
    export = tmp_path / "bofa.csv"
    export.write_text(
        "Date,Description,Amount,Running Bal.,Status,Transaction Type,Reference Number\n"
        "04/01/2026,COFFEE SHOP,-4.50,995.50,Posted,Debit,B123\n"
        "04/03/2026,DIRECT DEPOSIT,1000.00,1995.50,Posted,Credit,B124\n"
        "\n",
        encoding="utf-8",
    )

    result = BankOfAmericaTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    debit, credit = result.units
    assert debit.source_project == "bank_of_america_transactions_csv"
    assert debit.source_entity_type == "transaction"
    assert debit.metadata["date"] == "2026-04-01"
    assert debit.metadata["description"] == "COFFEE SHOP"
    assert debit.metadata["amount"] == -4.5
    assert debit.metadata["currency"] == "USD"
    assert debit.metadata["running_balance"] == 995.5
    assert debit.metadata["status"] == "Posted"
    assert debit.metadata["transaction_type"] == "Debit"
    assert debit.metadata["reference_number"] == "B123"
    assert debit.metadata["source_row"]["Running Bal."] == "995.50"
    assert debit.tags[:3] == ["finance", "transaction", "bank-of-america"]
    assert credit.metadata["amount"] == 1000.0
    assert credit.metadata["running_balance"] == 1995.5


def test_bank_of_america_transactions_csv_tolerates_preamble_and_blank_rows(tmp_path):
    export = tmp_path / "bofa.csv"
    export.write_text(
        "Bank of America Checking\n"
        "Downloaded,04/05/2026\n"
        "\n"
        "Date,Description,Amount,Running Bal.,Status,Transaction Type,Reference Number\n"
        "2026-04-01,COFFEE SHOP,($4.50),995.50,Posted,Debit,B123\n"
        "\n",
        encoding="utf-8",
    )

    first = BankOfAmericaTransactionsCsvAdapter(path=str(export)).ingest().units[0]
    second = BankOfAmericaTransactionsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("bank_of_america_transactions_csv:")
    assert first.metadata["amount"] == -4.5
