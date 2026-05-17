from __future__ import annotations

from graph.adapters.capital_one_transactions_csv import CapitalOneTransactionsCsvAdapter


def test_capital_one_transactions_csv_ingests_debit_and_credit_rows(tmp_path):
    export = tmp_path / "capital-one.csv"
    export.write_text(
        "Transaction Date,Posted Date,Card No.,Description,Category,Debit,Credit\n"
        "04/01/2026,04/02/2026,xxxx1234,COFFEE SHOP,Food & Drink,4.50,\n"
        "04/03/2026,04/04/2026,xxxx1234,RETURN,Shopping,,19.99\n"
        "\n",
        encoding="utf-8",
    )

    result = CapitalOneTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    debit, credit = result.units
    assert debit.source_project == "capital_one_transactions_csv"
    assert debit.source_entity_type == "transaction"
    assert debit.metadata["transaction_date"] == "2026-04-01"
    assert debit.metadata["posted_date"] == "2026-04-02"
    assert debit.metadata["card_number"] == "xxxx1234"
    assert debit.metadata["card_last_four"] == "1234"
    assert debit.metadata["description"] == "COFFEE SHOP"
    assert debit.metadata["category"] == "Food & Drink"
    assert debit.metadata["debit"] == 4.5
    assert debit.metadata["amount"] == -4.5
    assert debit.metadata["currency"] == "USD"
    assert debit.metadata["source_row"]["Debit"] == "4.50"
    assert debit.tags[:3] == ["finance", "transaction", "capital-one"]
    assert credit.metadata["credit"] == 19.99
    assert credit.metadata["amount"] == 19.99


def test_capital_one_transactions_csv_supports_amount_column_and_stable_ids(tmp_path):
    export = tmp_path / "capital-one.csv"
    export.write_text(
        "Transaction Date,Posted Date,Card No.,Description,Category,Amount\n"
        "2026-04-01,2026-04-02,1234,COFFEE SHOP,Food & Drink,($4.50)\n",
        encoding="utf-8",
    )

    first = CapitalOneTransactionsCsvAdapter(path=str(export)).ingest().units[0]
    second = CapitalOneTransactionsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("capital_one_transactions_csv:")
    assert first.metadata["amount"] == -4.5
