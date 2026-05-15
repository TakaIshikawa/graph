from __future__ import annotations

from graph.adapters.monzo_transactions_csv import MonzoTransactionsCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject


def test_monzo_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "monzo.csv"
    export.write_text(
        "Transaction ID,Date,Description,Merchant,Category,Amount,Currency,Local Amount,Local Currency,Notes,Tags,Address,City,Postcode,Country,Latitude,Longitude,Account ID,Account Name\n"
        "tx_1,2026-05-01T08:00:00Z,Coffee purchase,Cafe,Expenses,-3.50,GBP,-4.20,EUR,Flat white,\"coffee,work\",1 High St,London,EC1,GB,51.5,-0.1,acc_1,Current Account\n",
        encoding="utf-8",
    )

    result = MonzoTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.MONZO_TRANSACTIONS_CSV
    assert unit.source_id == "monzo_transactions_csv:tx_1"
    assert unit.metadata["timestamp"] == "2026-05-01T08:00:00+00:00"
    assert unit.metadata["description"] == "Coffee purchase"
    assert unit.metadata["merchant"] == "Cafe"
    assert unit.metadata["category"] == "Expenses"
    assert unit.metadata["amount"] == -3.5
    assert unit.metadata["currency"] == "GBP"
    assert unit.metadata["local_amount"] == -4.2
    assert unit.metadata["local_currency"] == "EUR"
    assert unit.metadata["notes"] == "Flat white"
    assert unit.metadata["tags"] == ["coffee", "work"]
    assert unit.metadata["address"]["city"] == "London"
    assert unit.metadata["address"]["latitude"] == 51.5
    assert unit.metadata["account_id"] == "acc_1"


def test_monzo_transactions_csv_tolerates_missing_optional_fields(tmp_path):
    export = tmp_path / "monzo.csv"
    export.write_text(
        "Date,Description,Amount,Currency,Merchant,Notes,Tags,Address\n"
        "2026-05-02,Transport,-2.80,GBP,,,,\n",
        encoding="utf-8",
    )

    result = MonzoTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["description"] == "Transport"
    assert unit.metadata["amount"] == -2.8
    assert "merchant" not in unit.metadata
    assert "notes" not in unit.metadata
    assert "tags" not in unit.metadata
    assert "address" not in unit.metadata


def test_monzo_transactions_csv_is_registered():
    assert "monzo_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("monzo-transactions-csv"), MonzoTransactionsCsvAdapter)
