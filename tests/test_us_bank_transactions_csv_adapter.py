from __future__ import annotations

from graph.adapters.us_bank_transactions_csv import UsBankTransactionsCsvAdapter


def test_us_bank_transactions_csv_ingests_bank_account_rows(tmp_path):
    export = tmp_path / "us-bank.csv"
    export.write_text(
        "Date,Transaction,Name,Memo,Amount,Balance,Category,Account,Reference Number\n"
        "04/01/2026,Debit,COFFEE SHOP,Latte,-4.50,995.50,Food & Drink,Checking,U100\n"
        "04/03/2026,Deposit,PAYROLL,April pay,2500.00,3495.50,Income,Checking,U101\n",
        encoding="utf-8",
    )

    result = UsBankTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    debit, deposit = result.units
    assert debit.source_project == "us_bank_transactions_csv"
    assert debit.source_entity_type == "transaction"
    assert debit.source_id == "us_bank_transactions_csv:U100"
    assert debit.metadata["date"] == "2026-04-01"
    assert debit.metadata["transaction"] == "Debit"
    assert debit.metadata["name"] == "COFFEE SHOP"
    assert debit.metadata["memo"] == "Latte"
    assert debit.metadata["amount"] == -4.5
    assert debit.metadata["balance"] == 995.5
    assert debit.metadata["category"] == "Food & Drink"
    assert debit.metadata["account"] == "Checking"
    assert debit.metadata["reference"] == "U100"
    assert debit.metadata["source_file"] == "us-bank.csv"
    assert debit.tags[:3] == ["finance", "transaction", "us-bank"]
    assert "COFFEE SHOP" in debit.title
    assert "Latte" in debit.content
    assert "-4.5 USD" in debit.content
    assert deposit.metadata["amount"] == 2500.0
    assert deposit.metadata["balance"] == 3495.5


def test_us_bank_transactions_csv_ingests_directory_and_card_like_unsigned_amounts(tmp_path):
    nested = tmp_path / "exports"
    nested.mkdir()
    export = nested / "us-bank.csv"
    export.write_text(
        "Date,Transaction,Name,Amount,Balance,Category,Account\n"
        "2026-04-01,Purchase,COFFEE SHOP,4.50,995.50,Food & Drink,Credit Card\n",
        encoding="utf-8",
    )

    first = UsBankTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]
    second = UsBankTransactionsCsvAdapter(path=str(tmp_path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("us_bank_transactions_csv:")
    assert first.metadata["amount"] == -4.5
    assert first.metadata["source_file"] == "us-bank.csv"
