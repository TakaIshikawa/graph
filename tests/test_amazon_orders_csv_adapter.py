from __future__ import annotations

from graph.adapters.amazon_orders_csv import AmazonOrdersCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation


def test_amazon_orders_csv_groups_orders_and_links_items(tmp_path):
    export = tmp_path / "amazon.csv"
    export.write_text("Order ID,Order Date,Shipment Date,Title,Category,ASIN,Quantity,Purchase Price Per Unit,Total Owed,Seller,Product URL\nA1,2026-05-01,2026-05-02,Book,Books,B001,1,10.00,10.00,Amazon,https://example.com/book\nA1,2026-05-01,2026-05-02,Cable,Electronics,C001,2,5.00,10.00,Shop,https://example.com/cable\n", encoding="utf-8")

    result = AmazonOrdersCsvAdapter(path=str(export)).ingest()

    assert len([unit for unit in result.units if unit.source_entity_type == "order"]) == 1
    assert len([unit for unit in result.units if unit.source_entity_type == "shipment"]) == 1
    assert len([unit for unit in result.units if unit.source_entity_type == "item"]) == 2
    assert len([unit for unit in result.units if unit.source_entity_type == "seller"]) == 2
    order = next(unit for unit in result.units if unit.source_entity_type == "order")
    assert order.metadata["total_owed"] == 20.0
    assert order.metadata["categories"] == {"Books": 1, "Electronics": 1}
    assert len(result.edges) == 5
    assert {edge.relation for edge in result.edges} == {EdgeRelation.CONTAINS}
    assert get_adapter("amazon_orders_csv", path=str(export)).name == "amazon_orders_csv"


def test_amazon_orders_csv_groups_shipments_and_preserves_unshipped_item_edges(tmp_path):
    export = tmp_path / "amazon.csv"
    export.write_text(
        "Order ID,Order Date,Shipment ID,Shipment Date,Tracking Number,Carrier,Title,Category,ASIN\n"
        "A1,2026-05-01,S1,2026-05-02,T1,UPS,Book,Books,B001\n"
        "A1,2026-05-01,S1,2026-05-02,T1,UPS,Cable,Electronics,C001\n"
        "A1,2026-05-01,,,,,Gift,Home,G001\n",
        encoding="utf-8",
    )

    result = AmazonOrdersCsvAdapter(path=str(export)).ingest()

    order = next(unit for unit in result.units if unit.source_entity_type == "order")
    shipment = next(unit for unit in result.units if unit.source_entity_type == "shipment")
    items = [unit for unit in result.units if unit.source_entity_type == "item"]
    unshipped = next(unit for unit in items if unit.title == "Gift")
    assert shipment.metadata["shipment_id"] == "S1"
    assert shipment.metadata["tracking_number"] == "T1"
    assert shipment.metadata["carrier"] == "UPS"
    assert shipment.metadata["item_count"] == 2
    assert (order.source_id, shipment.source_id) in {(edge.from_unit_id, edge.to_unit_id) for edge in result.edges}
    assert (shipment.source_id, unshipped.source_id) not in {(edge.from_unit_id, edge.to_unit_id) for edge in result.edges}
    assert (order.source_id, unshipped.source_id) in {(edge.from_unit_id, edge.to_unit_id) for edge in result.edges}


def test_amazon_orders_csv_emits_returns_with_refund_metadata_and_edges(tmp_path):
    export = tmp_path / "amazon.csv"
    export.write_text(
        "Order ID,Order Date,Title,ASIN,Return Date,Return Reason,Refund Amount,Return Status\n"
        "A1,2026-05-01,Book,B001,2026-05-05,Damaged,$10.50,Complete\n",
        encoding="utf-8",
    )

    result = AmazonOrdersCsvAdapter(path=str(export)).ingest(entity_types=["order", "item", "return"])

    order = next(unit for unit in result.units if unit.source_entity_type == "order")
    item = next(unit for unit in result.units if unit.source_entity_type == "item")
    return_unit = next(unit for unit in result.units if unit.source_entity_type == "return")
    assert return_unit.source_id.startswith("amazon_orders_csv:return:")
    assert return_unit.metadata["order_id"] == "A1"
    assert return_unit.metadata["item_source_id"] == item.source_id
    assert return_unit.metadata["asin"] == "B001"
    assert return_unit.metadata["return_date"] == "2026-05-05T00:00:00+00:00"
    assert return_unit.metadata["return_reason"] == "Damaged"
    assert return_unit.metadata["return_status"] == "Complete"
    assert return_unit.metadata["refund_amount"] == 10.5
    assert {(edge.from_unit_id, edge.to_unit_id) for edge in result.edges} == {
        (order.source_id, item.source_id),
        (order.source_id, return_unit.source_id),
        (item.source_id, return_unit.source_id),
    }


def test_amazon_orders_csv_return_filtering(tmp_path):
    export = tmp_path / "amazon.csv"
    export.write_text(
        "Order ID,Order Date,Title,Refunded Amount\n"
        "A1,2026-05-01,Book,5.00\n",
        encoding="utf-8",
    )

    return_only = AmazonOrdersCsvAdapter(path=str(export)).ingest(entity_types=["return"])
    item_only = AmazonOrdersCsvAdapter(path=str(export)).ingest(entity_types=["item"])

    assert [unit.source_entity_type for unit in return_only.units] == ["return"]
    assert return_only.edges == []
    assert [unit.source_entity_type for unit in item_only.units] == ["item"]
    assert item_only.edges == []


def test_amazon_orders_csv_emits_seller_aggregates_and_edges(tmp_path):
    first_export = tmp_path / "amazon1.csv"
    second_export = tmp_path / "amazon2.csv"
    first_export.write_text(
        "Order ID,Order Date,Title,Category,Total Owed,Seller\n"
        "A1,2026-05-01,Book,Books,10.00,Amazon\n"
        "A1,2026-05-01,Cable,Electronics,5.50,amazon\n",
        encoding="utf-8",
    )
    second_export.write_text(
        "Order ID,Order Date,Title,Category,Total Owed,Seller\n"
        "A2,2026-05-03,Pen,Office,3.25,Amazon\n"
        "A2,2026-05-03,Mug,Home,8.00,Shop\n",
        encoding="utf-8",
    )

    result = AmazonOrdersCsvAdapter(path=str(tmp_path)).ingest(entity_types=["seller", "item"])

    sellers = [unit for unit in result.units if unit.source_entity_type == "seller"]
    items = [unit for unit in result.units if unit.source_entity_type == "item"]
    assert len(sellers) == 2

    amazon = next(unit for unit in sellers if unit.metadata["normalized_seller"] == "amazon")
    amazon_items = [item for item in items if str(item.metadata.get("seller", "")).casefold() == "amazon"]
    assert amazon.source_id.startswith("amazon_orders_csv:seller:")
    assert amazon.metadata["seller"] == "Amazon"
    assert amazon.metadata["item_count"] == 3
    assert amazon.metadata["order_count"] == 2
    assert amazon.metadata["total_owed"] == 18.75
    assert amazon.metadata["categories"] == {"Books": 1, "Electronics": 1, "Office": 1}
    assert amazon.metadata["order_source_ids"] == ["amazon_orders_csv:A1", "amazon_orders_csv:A2"]
    assert amazon.metadata["item_source_ids"] == sorted(item.source_id for item in amazon_items)

    seller_edges = [edge for edge in result.edges if edge.from_unit_id == amazon.source_id]
    assert {edge.to_unit_id for edge in seller_edges} == {item.source_id for item in amazon_items}
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in seller_edges)
    assert all(edge.metadata["relation_type"] == "seller_contains_item" for edge in seller_edges)


def test_amazon_orders_csv_seller_filtering(tmp_path):
    export = tmp_path / "amazon.csv"
    export.write_text(
        "Order ID,Order Date,Title,Seller\n"
        "A1,2026-05-01,Book,Amazon\n",
        encoding="utf-8",
    )

    seller_only = AmazonOrdersCsvAdapter(path=str(export)).ingest(entity_types=["seller"])
    item_only = AmazonOrdersCsvAdapter(path=str(export)).ingest(entity_types=["item"])

    assert [unit.source_entity_type for unit in seller_only.units] == ["seller"]
    assert seller_only.edges == []
    assert [unit.source_entity_type for unit in item_only.units] == ["item"]
    assert item_only.edges == []


def test_amazon_orders_csv_reports_return_entity_type():
    assert AmazonOrdersCsvAdapter().entity_types == ["order", "shipment", "item", "return", "seller"]
