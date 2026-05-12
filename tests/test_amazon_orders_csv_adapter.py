from __future__ import annotations

from graph.adapters.amazon_orders_csv import AmazonOrdersCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation


def test_amazon_orders_csv_groups_orders_and_links_items(tmp_path):
    export = tmp_path / "amazon.csv"
    export.write_text("Order ID,Order Date,Shipment Date,Title,Category,ASIN,Quantity,Purchase Price Per Unit,Total Owed,Seller,Product URL\nA1,2026-05-01,2026-05-02,Book,Books,B001,1,10.00,10.00,Amazon,https://example.com/book\nA1,2026-05-01,2026-05-02,Cable,Electronics,C001,2,5.00,10.00,Shop,https://example.com/cable\n", encoding="utf-8")

    result = AmazonOrdersCsvAdapter(path=str(export)).ingest()

    assert len([unit for unit in result.units if unit.source_entity_type == "order"]) == 1
    assert len([unit for unit in result.units if unit.source_entity_type == "item"]) == 2
    order = next(unit for unit in result.units if unit.source_entity_type == "order")
    assert order.metadata["total_owed"] == 20.0
    assert order.metadata["categories"] == {"Books": 1, "Electronics": 1}
    assert len(result.edges) == 2
    assert {edge.relation for edge in result.edges} == {EdgeRelation.CONTAINS}
    assert get_adapter("amazon_orders_csv", path=str(export)).name == "amazon_orders_csv"
