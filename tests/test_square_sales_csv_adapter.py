from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.square_sales_csv import SquareSalesCsvAdapter
from graph.types.models import SyncState


def test_square_sales_csv_ingests_sales_rows(tmp_path):
    export = tmp_path / "square.csv"
    export.write_text(
        "Transaction ID,Receipt Number,Date,Location,Customer,Item,Variation,Category,SKU,Quantity,Gross Sales,Discounts,Tax,Tip,Net Total,Payment Method,Currency\n"
        "txn1,R001,2026-05-01 10:30:00,Downtown,Ada,Latte,Large,Drinks,SKU1,2,10.00,1.00,0.80,2.00,11.80,Visa,USD\n",
        encoding="utf-8",
    )

    unit = SquareSalesCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "square_sales_csv"
    assert unit.source_id == "square_sales_csv:txn1"
    assert unit.source_entity_type == "sale"
    assert unit.metadata["timestamp"] == "2026-05-01T10:30:00+00:00"
    assert unit.metadata["receipt_number"] == "R001"
    assert unit.metadata["item"] == "Latte"
    assert unit.metadata["quantity"] == 2.0
    assert unit.metadata["gross_sales"] == 10.0
    assert unit.metadata["discounts"] == 1.0
    assert unit.metadata["tax"] == 0.8
    assert unit.metadata["tip"] == 2.0
    assert unit.metadata["net_total"] == 11.8
    assert {"square", "sale", "Downtown", "Drinks", "Visa"}.issubset(set(unit.tags))
    assert "Net total: 11.8 USD" in unit.content


def test_square_sales_csv_aliases_directory_since_and_entity_filter(tmp_path):
    (tmp_path / "old.csv").write_text("Date Time,Receipt #,Item Name,Total\n2026-05-01,OLD,Old item,1\n", encoding="utf-8")
    (tmp_path / "new.csv").write_text(
        "Payment Time,Payment ID,Receipt,Location Name,Description,Qty,Gross Amount,Discount Amount,Taxes,Tips,Net Sales,Tender Type,Currency Code\n"
        "2026-05-03,PAY1,R002,Uptown,Bagel,3,9.00,0,0.72,1.50,11.22,Cash,USD\n",
        encoding="utf-8",
    )

    adapter = SquareSalesCsvAdapter(path=str(tmp_path))
    sync = SyncState(source_project="square_sales_csv", source_entity_type="sale", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.metadata["item"] for unit in first.units] == ["Bagel"]
    assert first.units[0].source_id == "square_sales_csv:PAY1"
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["order"]).units == []


def test_square_sales_csv_sparse_rows(tmp_path):
    export = tmp_path / "sparse.csv"
    export.write_text("Receipt Number,Item,Net Total\n, ,\nR003,Scone,4.25\n", encoding="utf-8")

    result = SquareSalesCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id == "square_sales_csv:R003"
    assert result.units[0].metadata["net_total"] == 4.25
