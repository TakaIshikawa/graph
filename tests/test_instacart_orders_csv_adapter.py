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
