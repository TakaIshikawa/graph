from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.amazon_orders_csv import AmazonOrdersCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_amazon_orders_csv_ingests_order_item_metadata(tmp_path):
    export = tmp_path / "amazon.csv"
    export.write_text(
        "Order ID,Order Date,Title,ASIN/ISBN,Category,Seller,Quantity,Item Total,Currency,Shipment Date,Tracking Number,Ship To,URL,Order Status\n"
        "A1,2026-05-01,Book,B001,Books,Amazon,1,$10.50,USD,2026-05-02,T1,Ada,https://example.com/book,Shipped\n",
        encoding="utf-8",
    )

    result = AmazonOrdersCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.AMAZON_ORDERS_CSV
    assert unit.source_entity_type == "order_item"
    assert unit.metadata["order_id"] == "A1"
    assert unit.metadata["order_date"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["product_title"] == "Book"
    assert unit.metadata["asin_isbn"] == "B001"
    assert unit.metadata["seller"] == "Amazon"
    assert unit.metadata["quantity"] == 1
    assert unit.metadata["amount"] == 10.5
    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["order_status"] == "Shipped"
    assert unit.metadata["tracking_number"] == "T1"
    assert unit.metadata["ship_to"] == "Ada"
    assert unit.metadata["url"] == "https://example.com/book"
    assert unit.metadata["source_file"] == "amazon.csv"
    assert {"amazon", "order_item", "Books", "Amazon", "Shipped"}.issubset(set(unit.tags))


def test_amazon_orders_csv_directory_bom_invalid_rows_and_stable_ids(tmp_path):
    first_export = tmp_path / "first.csv"
    second_export = tmp_path / "second.csv"
    first_export.write_text(
        "\ufeffOrder ID,Order Date,Title,ASIN,Item Total\n"
        "A1,2026-05-01,Book,B001,10.00\n"
        "A2,2026-05-02,,B002,20.00\n",
        encoding="utf-8",
    )
    second_export.write_text(
        "Order ID,Order Date,Product Name,ISBN,Amount\n"
        "A3,2026-05-03,Cable,C001,5.25\n",
        encoding="utf-8",
    )

    first_result = AmazonOrdersCsvAdapter(path=str(tmp_path)).ingest(entity_types=["order_item"])
    second_result = AmazonOrdersCsvAdapter(path=str(tmp_path)).ingest(entity_types=["order_item"])

    assert sorted(unit.title for unit in first_result.units) == ["Book", "Cable"]
    assert [unit.source_id for unit in second_result.units] == [unit.source_id for unit in first_result.units]
    book = next(unit for unit in first_result.units if unit.title == "Book")
    assert book.metadata["source_file"] == "first.csv"


def test_amazon_orders_csv_filters_entity_types_and_since(tmp_path):
    export = tmp_path / "amazon.csv"
    export.write_text(
        "Order ID,Order Date,Shipment Date,Title\n"
        "old,2026-04-01,2026-04-02,Old item\n"
        "new,2026-05-01,2026-05-03,New item\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="amazon_orders_csv",
        source_entity_type="order_item",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = AmazonOrdersCsvAdapter(path=str(export)).ingest(since=since, entity_types=["order_item"])

    assert [unit.title for unit in result.units] == ["New item"]
    assert AmazonOrdersCsvAdapter(path=str(export)).ingest(entity_types=["order"]).units == []
    assert get_adapter("amazon_orders_csv", path=str(export)).name == "amazon_orders_csv"
