from __future__ import annotations

from graph.adapters.bank_of_america_transactions_csv import BankOfAmericaTransactionsCsvAdapter


def test_bank_of_america_transactions_csv_ingests_rows_with_balances(tmp_path):
    export = tmp_path / "bofa.csv"
    export.write_text(
        "Posted Date,Reference Number,Payee,Address,Amount,Running Bal.,Category,Account,Description,Memo\n"
        "04/01/2026,B123,COFFEE SHOP,1 Main St,-4.50,995.50,Food,Checking,Card purchase,Latte\n"
        "04/03/2026,B124,DIRECT DEPOSIT,,1000.00,1995.50,Income,Checking,Payroll,\n"
        "\n",
        encoding="utf-8",
    )

    result = BankOfAmericaTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    debit, credit = result.units
    assert debit.source_project == "bank_of_america_transactions_csv"
    assert debit.source_entity_type == "transaction"
    assert debit.source_id == "bank_of_america_transactions_csv:B123"
    assert debit.metadata["date"] == "2026-04-01"
    assert debit.metadata["posted_date"] == "2026-04-01"
    assert debit.metadata["reference_number"] == "B123"
    assert debit.metadata["payee"] == "COFFEE SHOP"
    assert debit.metadata["address"] == "1 Main St"
    assert debit.metadata["description"] == "Card purchase"
    assert debit.metadata["memo"] == "Latte"
    assert debit.metadata["amount"] == -4.5
    assert debit.metadata["currency"] == "USD"
    assert debit.metadata["running_balance"] == 995.5
    assert debit.metadata["account"] == "Checking"
    assert debit.metadata["category"] == "Food"
    assert debit.metadata["source_row"]["Running Bal."] == "995.50"
    assert debit.metadata["source_file"] == "bofa.csv"
    assert debit.tags[:3] == ["finance", "transaction", "bank-of-america"]
    assert "Card purchase" in debit.title
    assert "COFFEE SHOP" in debit.content
    assert credit.metadata["amount"] == 1000.0
    assert credit.metadata["running_balance"] == 1995.5


def test_bank_of_america_transactions_csv_ingests_directories_and_skips_malformed_csvs(tmp_path):
    nested = tmp_path / "exports"
    nested.mkdir()
    malformed = tmp_path / "bad.csv"
    malformed.write_bytes(b"\xff\xfe\x00")
    export = nested / "bofa.csv"
    export.write_text(
        "Bank of America Checking\n"
        "Downloaded,04/05/2026\n"
        "\n"
        "Posted Date,Reference Number,Payee,Address,Amount,Running Bal.,Category,Account,Description,Memo\n"
        "2026-04-01,,COFFEE SHOP,1 Main St,($4.50),995.50,Food,Checking,Card purchase,Latte\n"
        "\n",
        encoding="utf-8",
    )

    first = BankOfAmericaTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]
    second = BankOfAmericaTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("bank_of_america_transactions_csv:")
    assert first.metadata["amount"] == -4.5
    assert first.metadata["source_file"] == "bofa.csv"
