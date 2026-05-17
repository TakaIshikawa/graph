from __future__ import annotations

from graph.adapters.instacart_orders_csv import InstacartOrdersCsvAdapter
from graph.adapters.registry import get_adapter


def test_instacart_orders_csv_groups_rows_into_orders(tmp_path):
    export = tmp_path / "instacart.csv"
    export.write_text("order_id,ordered_at,store_name,item_name,quantity,unit_price,total_price,department\nord-1,2026-05-01,Market,Apples,2,1.00,2.00,Produce\nord-1,2026-05-01,Market,Milk,1,3.50,3.50,Dairy\n", encoding="utf-8")

    result = InstacartOrdersCsvAdapter(path=str(export)).ingest()

    unit = result.units[0]
    assert unit.source_id == "instacart_orders_csv:ord-1"
    assert unit.metadata["item_count"] == 2
    assert unit.metadata["total_price"] == 5.5
    assert unit.metadata["departments"] == {"Dairy": 1, "Produce": 1}
    assert [item["name"] for item in unit.metadata["items"]] == ["Apples", "Milk"]
    assert get_adapter("instacart_orders_csv", path=str(export)).name == "instacart_orders_csv"


def test_instacart_orders_csv_preserves_order_fees_and_delivery_window(tmp_path):
    export = tmp_path / "instacart.csv"
    export.write_text(
        "order_id,ordered_at,store_name,item_name,quantity,total_price,department,subtotal,tax,"
        "service_fee,delivery_fee,tip,discount,total_charged,shopper_name,delivery_start,"
        "delivery_end,delivery_address\n"
        "ord-2,2026-05-02,Market,Bread,1,4.00,Bakery,10.00,0.85,1.50,3.99,5.00,-2.00,"
        "19.34,Ada,2026-05-02 13:00,2026-05-02 15:00,123 Market St\n"
        "ord-2,2026-05-02,Market,Eggs,1,6.00,Dairy,10.00,0.85,1.50,3.99,5.00,-2.00,"
        "19.34,Ada,2026-05-02 13:00,2026-05-02 15:00,123 Market St\n",
        encoding="utf-8",
    )

    result = InstacartOrdersCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["subtotal"] == 10.0
    assert unit.metadata["tax"] == 0.85
    assert unit.metadata["service_fee"] == 1.5
    assert unit.metadata["delivery_fee"] == 3.99
    assert unit.metadata["tip"] == 5.0
    assert unit.metadata["discount"] == -2.0
    assert unit.metadata["total_charged"] == 19.34
    assert unit.metadata["shopper_name"] == "Ada"
    assert unit.metadata["delivery_start"] == "2026-05-02T13:00:00+00:00"
    assert unit.metadata["delivery_end"] == "2026-05-02T15:00:00+00:00"
    assert unit.metadata["delivery_address"] == "123 Market St"
    assert unit.metadata["total_price"] == 10.0
    assert [item["name"] for item in unit.metadata["items"]] == ["Bread", "Eggs"]
