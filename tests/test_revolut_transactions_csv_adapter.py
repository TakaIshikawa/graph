from __future__ import annotations

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.revolut_transactions_csv import RevolutTransactionsCsvAdapter
from graph.types.enums import SourceProject


def test_revolut_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "revolut.csv"
    export.write_text(
        "Completed Date,Started Date,Description,Paid Out,Paid In,Exchange Out,Exchange In,Balance,Currency,Category,Account,Transaction ID\n"
        "2026-05-01 08:00:00,2026-05-01 07:59:00,Coffee,3.50,,4.10,,96.50,GBP,Food,Personal,REV1\n",
        encoding="utf-8",
    )

    result = RevolutTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.REVOLUT_TRANSACTIONS_CSV
    assert unit.source_id == "revolut_transactions_csv:REV1"
    assert unit.metadata["completed_at"] == "2026-05-01T08:00:00+00:00"
    assert unit.metadata["started_at"] == "2026-05-01T07:59:00+00:00"
    assert unit.metadata["description"] == "Coffee"
    assert unit.metadata["paid_out"] == 3.5
    assert unit.metadata["amount"] == -3.5
    assert unit.metadata["exchange_out"] == 4.1
    assert unit.metadata["balance"] == 96.5
    assert unit.metadata["currency"] == "GBP"
    assert unit.metadata["category"] == "Food"
    assert unit.metadata["account_name"] == "Personal"


def test_revolut_transactions_csv_uses_started_date_and_skips_blank_rows(tmp_path):
    export = tmp_path / "revolut.csv"
    export.write_text(
        "Started Date,Description,Paid In,Currency\n"
        ",,,\n"
        "2026-05-02,Salary,2500,GBP\n",
        encoding="utf-8",
    )

    result = RevolutTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["started_at"] == "2026-05-02T00:00:00+00:00"
    assert unit.metadata["amount"] == 2500.0


def test_revolut_transactions_csv_is_registered():
    assert "revolut_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("revolut-transactions-csv"), RevolutTransactionsCsvAdapter)
