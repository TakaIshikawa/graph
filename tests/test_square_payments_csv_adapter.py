from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.square_payments_csv import SquarePaymentsCsvAdapter
from graph.types.models import SyncState


def test_square_payments_csv_ingests_payment_rows(tmp_path):
    export = tmp_path / "payments.csv"
    export.write_text(
        "Date,Time,Timezone,Gross Sales,Discounts,Net Sales,Tax,Tip,Fees,Total Collected,Payment ID,Order ID,Customer ID,Customer Name,Card Brand,Last 4,Currency\n"
        "2026-05-01,10:30:00,UTC,20.00,1.00,19.00,1.52,3.00,-0.75,22.77,pay_1,ord_1,cus_1,Ada Lovelace,Visa,1234,USD\n",
        encoding="utf-8",
    )

    unit = SquarePaymentsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "square_payments_csv"
    assert unit.source_id == "square_payments_csv:pay_1"
    assert unit.source_entity_type == "payment"
    assert unit.metadata["timestamp"] == "2026-05-01T10:30:00+00:00"
    assert unit.metadata["timezone"] == "UTC"
    assert unit.metadata["gross_sales"] == 20.0
    assert unit.metadata["discounts"] == 1.0
    assert unit.metadata["net_sales"] == 19.0
    assert unit.metadata["tax"] == 1.52
    assert unit.metadata["tip"] == 3.0
    assert unit.metadata["fees"] == -0.75
    assert unit.metadata["total_collected"] == 22.77
    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["order_id"] == "ord_1"
    assert unit.metadata["customer_id"] == "cus_1"
    assert unit.metadata["customer_name"] == "Ada Lovelace"
    assert unit.metadata["card_brand"] == "Visa"
    assert unit.metadata["last_4"] == "1234"
    assert unit.metadata["source_file"] == "payments.csv"
    assert unit.metadata["source_row"]["Payment ID"] == "pay_1"
    assert {"square", "payment", "Visa", "Ada Lovelace"}.issubset(set(unit.tags))
    assert "Total collected: 22.77 USD" in unit.content


def test_square_payments_csv_directory_since_ids_and_entity_filter(tmp_path):
    (tmp_path / "old.csv").write_text(
        "Date Time,Payment ID,Customer Name,Total Collected\n2026-05-01 09:00:00,old,Ada,1.00\n",
        encoding="utf-8",
    )
    (tmp_path / "new.csv").write_text(
        "Timestamp,Order ID,Customer,Total,Card Brand,Card Last Four,Processing Fees\n"
        "2026-05-03 12:00:00,ord_2,Grace Hopper,$15.25,Mastercard,card ending 9876,($0.55)\n",
        encoding="utf-8",
    )

    adapter = SquarePaymentsCsvAdapter(path=str(tmp_path))
    sync = SyncState(
        source_project="square_payments_csv",
        source_entity_type="payment",
        last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert len(first.units) == 1
    assert first.units[0].source_id == "square_payments_csv:ord_2"
    assert first.units[0].metadata["customer_name"] == "Grace Hopper"
    assert first.units[0].metadata["total_collected"] == 15.25
    assert first.units[0].metadata["fees"] == -0.55
    assert first.units[0].metadata["last_4"] == "9876"
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["sale"]).units == []


def test_square_payments_csv_sparse_rows(tmp_path):
    export = tmp_path / "sparse.csv"
    export.write_text(
        "Payment ID,Customer Name,Total Collected\n"
        ", ,\n"
        ",Sparse Customer,4.25\n",
        encoding="utf-8",
    )

    result = SquarePaymentsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["customer_name"] == "Sparse Customer"
    assert result.units[0].metadata["total_collected"] == 4.25
    assert result.units[0].source_id.startswith("square_payments_csv:")
