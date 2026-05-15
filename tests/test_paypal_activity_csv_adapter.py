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


def test_paypal_activity_csv_is_registered():
    assert "paypal_activity_csv" in list_adapters()
    assert isinstance(get_adapter("paypal-activity-csv"), PaypalActivityCsvAdapter)
