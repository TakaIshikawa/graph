from __future__ import annotations

from graph.adapters.paypal_activity_csv import PaypalActivityCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject


def test_paypal_activity_csv_ingests_activity_rows(tmp_path):
    export = tmp_path / "paypal.csv"
    export.write_text(
        "Date,Time,Transaction ID,Name,Type,Status,Subject,Gross,Fee,Net,Currency,Balance,Reference Txn ID\n"
        "2026-05-01,10:30:00,TXN1,Ada,Payment,Completed,Consulting,\"$120.00\",\"-$4.00\",\"$116.00\",USD,\"$500.00\",REF1\n",
        encoding="utf-8",
    )

    result = PaypalActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.PAYPAL_ACTIVITY_CSV
    assert unit.source_id == "paypal_activity_csv:TXN1"
    assert unit.metadata["timestamp"] == "2026-05-01T10:30:00+00:00"
    assert unit.metadata["name"] == "Ada"
    assert unit.metadata["type"] == "Payment"
    assert unit.metadata["status"] == "Completed"
    assert unit.metadata["subject"] == "Consulting"
    assert unit.metadata["gross"] == 120.0
    assert unit.metadata["fee"] == -4.0
    assert unit.metadata["net"] == 116.0
    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["balance_impact"] == 116.0
    assert unit.metadata["reference"] == "REF1"


def test_paypal_activity_csv_handles_blank_fee_net_and_negative_parentheses(tmp_path):
    export = tmp_path / "paypal.csv"
    export.write_text(
        "Date,Transaction ID,Name,Type,Status,Note,Gross,Fee,Net,Currency\n"
        "2026-05-02,TXN2,Grace,Refund,Completed,Refund note,(25.50),,,USD\n",
        encoding="utf-8",
    )

    result = PaypalActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["gross"] == -25.5
    assert unit.metadata["balance_impact"] == -25.5
    assert "fee" not in unit.metadata
    assert "net" not in unit.metadata
    assert unit.metadata["note"] == "Refund note"


def test_paypal_activity_csv_preserves_shipping_and_payer_metadata(tmp_path):
    export = tmp_path / "paypal.csv"
    export.write_text(
        "Date,Transaction ID,Name,Type,Status,Gross,Currency,Payer Email,Shipping Name,Shipping Address,"
        "Shipping City,Shipping State,Shipping Postal Code,Shipping Country,Item ID,Item URL,"
        "Protection Eligibility,Shipping Address 2\n"
        "2026-05-03,TXN3,Lin,Payment,Completed,$42.00,USD,lin@example.com,Lin Chen,"
        "123 Market St,San Francisco,CA,94105,US,SKU-7,https://example.com/items/sku-7,"
        "Eligible,\n",
        encoding="utf-8",
    )

    result = PaypalActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id == "paypal_activity_csv:TXN3"
    assert unit.metadata["payer_email"] == "lin@example.com"
    assert unit.metadata["shipping_name"] == "Lin Chen"
    assert unit.metadata["shipping_address"] == "123 Market St"
    assert unit.metadata["shipping_city"] == "San Francisco"
    assert unit.metadata["shipping_state"] == "CA"
    assert unit.metadata["shipping_postal_code"] == "94105"
    assert unit.metadata["shipping_country"] == "US"
    assert unit.metadata["item_id"] == "SKU-7"
    assert unit.metadata["item_url"] == "https://example.com/items/sku-7"
    assert unit.metadata["protection_eligibility"] == "Eligible"
    assert "shipping_address_2" not in unit.metadata


def test_paypal_activity_csv_emits_counterparty_aggregates_and_edges(tmp_path):
    export = tmp_path / "paypal.csv"
    export.write_text(
        "Date,Transaction ID,Name,From Email Address,To Email Address,Type,Status,Gross,Net,Currency\n"
        "2026-05-01,TXN1,Ada,ada@example.com,,Payment,Completed,120,116,USD\n"
        "2026-05-03,TXN2, ada ,,ada@example.com,Refund,Pending,(25.50),,USD\n"
        "2026-05-04,TXN3,,,,Fee,Completed,-1,-1,USD\n",
        encoding="utf-8",
    )

    result = PaypalActivityCsvAdapter(path=str(export)).ingest(entity_types=["activity", "counterparty"])

    counterparties = [unit for unit in result.units if unit.source_entity_type == "counterparty"]
    assert len(counterparties) == 1
    counterparty = counterparties[0]
    assert counterparty.metadata["counterparty"] == "Ada"
    assert counterparty.metadata["activity_count"] == 2
    assert counterparty.metadata["first_seen"] == "2026-05-01T00:00:00+00:00"
    assert counterparty.metadata["last_seen"] == "2026-05-03T00:00:00+00:00"
    assert counterparty.metadata["net_total"] == 90.5
    assert counterparty.metadata["currencies"] == ["USD"]
    assert counterparty.metadata["statuses"] == ["Completed", "Pending"]
    assert {(edge.from_unit_id, edge.to_unit_id, edge.metadata["relation_type"]) for edge in result.edges} == {
        ("paypal_activity_csv:TXN1", counterparty.source_id, "activity_counterparty"),
        ("paypal_activity_csv:TXN2", counterparty.source_id, "activity_counterparty"),
    }

    assert [unit.source_entity_type for unit in PaypalActivityCsvAdapter(path=str(export)).ingest(entity_types=["counterparty"]).units] == ["counterparty"]
    assert PaypalActivityCsvAdapter(path=str(export)).ingest(entity_types=["activity"]).edges == []


def test_paypal_activity_csv_is_registered():
    assert "paypal_activity_csv" in list_adapters()
    assert isinstance(get_adapter("paypal-activity-csv"), PaypalActivityCsvAdapter)
