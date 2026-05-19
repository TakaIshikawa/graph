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


def test_monzo_transactions_csv_emits_merchant_aggregates_and_edges(tmp_path):
    export = tmp_path / "monzo.csv"
    export.write_text(
        "Transaction ID,Date,Description,Merchant,Category,Amount,Currency\n"
        "tx_1,2026-05-01,Coffee,Cafe,Expenses,-3.50,GBP\n"
        "tx_2,2026-05-03,Lunch,  cafe  ,Expenses,-8.25,GBP\n"
        "tx_3,2026-05-04,Transfer,,Transfers,20.00,GBP\n",
        encoding="utf-8",
    )

    result = MonzoTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["transaction", "merchant"])

    merchants = [unit for unit in result.units if unit.source_entity_type == "merchant"]
    assert len(merchants) == 1
    merchant = merchants[0]
    assert merchant.metadata["merchant"] == "Cafe"
    assert merchant.metadata["normalized_merchant"] == "cafe"
    assert merchant.metadata["transaction_count"] == 2
    assert merchant.metadata["first_seen"] == "2026-05-01T00:00:00+00:00"
    assert merchant.metadata["last_seen"] == "2026-05-03T00:00:00+00:00"
    assert merchant.metadata["total_amount"] == -11.75
    assert merchant.metadata["currency"] == "GBP"
    assert merchant.metadata["transaction_source_ids"] == ["monzo_transactions_csv:tx_1", "monzo_transactions_csv:tx_2"]
    assert {(edge.from_unit_id, edge.to_unit_id, edge.metadata["relation_type"]) for edge in result.edges} == {
        ("monzo_transactions_csv:tx_1", merchant.source_id, "transaction_merchant"),
        ("monzo_transactions_csv:tx_2", merchant.source_id, "transaction_merchant"),
    }

    transaction_only = MonzoTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["transaction"])
    merchant_only = MonzoTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["merchant"])

    assert [unit.source_entity_type for unit in transaction_only.units] == ["transaction", "transaction", "transaction"]
    assert [unit.source_entity_type for unit in merchant_only.units] == ["merchant"]
    assert transaction_only.edges == []
    assert merchant_only.edges == []


def test_monzo_transactions_csv_is_registered():
    assert "monzo_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("monzo-transactions-csv"), MonzoTransactionsCsvAdapter)
