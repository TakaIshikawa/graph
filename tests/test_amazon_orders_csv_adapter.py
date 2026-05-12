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
    order = next(unit for unit in result.units if unit.source_entity_type == "order")
    assert order.metadata["total_owed"] == 20.0
    assert order.metadata["categories"] == {"Books": 1, "Electronics": 1}
    assert len(result.edges) == 3
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


def test_amazon_orders_csv_emits_returns_and_links_to_items(tmp_path):
    export = tmp_path / "amazon.csv"
    export.write_text(
        "Order ID,Order Date,Title,ASIN,Return Date,Return Reason,Refund Amount,Return Status\n"
        "A1,2026-05-01,Book,B001,2026-05-05,Damaged,$10.25,Completed\n",
        encoding="utf-8",
    )

    result = AmazonOrdersCsvAdapter(path=str(export)).ingest()

    item = next(unit for unit in result.units if unit.source_entity_type == "item")
    returned = next(unit for unit in result.units if unit.source_entity_type == "return")
    assert returned.metadata["order_id"] == "A1"
    assert returned.metadata["item_source_id"] == item.source_id
    assert returned.metadata["refund_amount"] == 10.25
    assert returned.metadata["return_date"] == "2026-05-05T00:00:00+00:00"
    assert returned.metadata["return_reason"] == "Damaged"
    assert returned.metadata["return_status"] == "Completed"
    assert (item.source_id, returned.source_id) in {
        (edge.from_unit_id, edge.to_unit_id) for edge in result.edges
    }


def test_amazon_orders_csv_return_filtering(tmp_path):
    export = tmp_path / "amazon.csv"
    export.write_text(
        "Order ID,Order Date,Title,Refunded Amount\nA1,2026-05-01,Book,5.00\n",
        encoding="utf-8",
    )

    result = AmazonOrdersCsvAdapter(path=str(export)).ingest(entity_types=["return"])

    assert [unit.source_entity_type for unit in result.units] == ["return"]
    assert result.edges == []
