from __future__ import annotations

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.stripe_balance_transactions_csv import StripeBalanceTransactionsCsvAdapter
from graph.types.enums import SourceProject


def test_stripe_balance_transactions_csv_ingests_rows(tmp_path):
    export = tmp_path / "stripe.csv"
    export.write_text(
        "Created,Available On,id,Type,Description,Amount,Fee,Net,Currency,Source,Payout,Reporting Category,Status\n"
        "2026-05-01T08:00:00Z,2026-05-03,txn_1,charge,Payment from Ada,1000,35,965,usd,ch_1,po_1,charge,succeeded\n",
        encoding="utf-8",
    )

    result = StripeBalanceTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.STRIPE_BALANCE_TRANSACTIONS_CSV
    assert unit.source_id == "stripe_balance_transactions_csv:txn_1"
    assert unit.metadata["created_at"] == "2026-05-01T08:00:00+00:00"
    assert unit.metadata["available_at"] == "2026-05-03T00:00:00+00:00"
    assert unit.metadata["type"] == "charge"
    assert unit.metadata["description"] == "Payment from Ada"
    assert unit.metadata["amount"] == 1000.0
    assert unit.metadata["fee"] == 35.0
    assert unit.metadata["net"] == 965.0
    assert unit.metadata["currency"] == "usd"
    assert unit.metadata["source"] == "ch_1"
    assert unit.metadata["payout"] == "po_1"
    assert unit.metadata["reporting_category"] == "charge"
    assert unit.metadata["status"] == "succeeded"


def test_stripe_balance_transactions_csv_includes_status_type_reporting_content(tmp_path):
    export = tmp_path / "stripe.csv"
    export.write_text(
        "Created,Type,Reporting Category,Status,Amount,Currency\n"
        "2026-05-02,refund,refund,pending,-100,usd\n",
        encoding="utf-8",
    )

    result = StripeBalanceTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert "refund" in result.units[0].content
    assert "pending" in result.units[0].content


def test_stripe_balance_transactions_csv_emits_reporting_category_aggregates_and_edges(tmp_path):
    export = tmp_path / "stripe.csv"
    export.write_text(
        "Created,id,Type,Reporting Category,Status,Amount,Fee,Net,Currency\n"
        "2026-05-01,txn_1,charge,charge,succeeded,1000,35,965,usd\n"
        "2026-05-03,txn_2,charge, charge ,pending,500,20,480,usd\n"
        "2026-05-04,txn_3,adjustment,,succeeded,100,0,100,usd\n",
        encoding="utf-8",
    )

    result = StripeBalanceTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["transaction", "reporting_category"])

    categories = [unit for unit in result.units if unit.source_entity_type == "reporting_category"]
    assert len(categories) == 1
    category = categories[0]
    assert category.metadata["reporting_category"] == "charge"
    assert category.metadata["transaction_count"] == 2
    assert category.metadata["amount_total"] == 1500.0
    assert category.metadata["fee_total"] == 55.0
    assert category.metadata["net_total"] == 1445.0
    assert category.metadata["currencies"] == ["usd"]
    assert category.metadata["first_seen"] == "2026-05-01T00:00:00+00:00"
    assert category.metadata["last_seen"] == "2026-05-03T00:00:00+00:00"
    assert category.metadata["statuses"] == ["pending", "succeeded"]
    assert {(edge.from_unit_id, edge.to_unit_id, edge.metadata["relation_type"]) for edge in result.edges} == {
        ("stripe_balance_transactions_csv:txn_1", category.source_id, "transaction_reporting_category"),
        ("stripe_balance_transactions_csv:txn_2", category.source_id, "transaction_reporting_category"),
    }

    assert [unit.source_entity_type for unit in StripeBalanceTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["reporting_category"]).units] == ["reporting_category"]
    assert StripeBalanceTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["transaction"]).edges == []


def test_stripe_balance_transactions_csv_is_registered():
    assert "stripe_balance_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("stripe-balance-transactions-csv"), StripeBalanceTransactionsCsvAdapter)
