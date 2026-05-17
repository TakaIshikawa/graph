from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.shopify_orders_csv import ShopifyOrdersCsvAdapter
from graph.types.models import SyncState


def test_shopify_orders_csv_ingests_order_rows(tmp_path):
    export = tmp_path / "shopify.csv"
    export.write_text(
        "Name,Created at,Paid at,Fulfilled at,Financial Status,Fulfillment Status,Currency,Subtotal,Shipping,Taxes,Total,Discount Amount,Lineitem name,Lineitem quantity,Lineitem price,SKU,Customer name,Customer email,Billing country,Shipping country,Tags,Note\n"
        "#1001,2026-05-01 10:00:00,2026-05-01 10:05:00,2026-05-02 09:00:00,paid,fulfilled,USD,40.00,5.00,3.60,48.60,0,T-Shirt,2,20.00,TSHIRT-1,Ada Lovelace,ada@example.com,US,US,\"vip, wholesale\",Gift note\n",
        encoding="utf-8",
    )

    unit = ShopifyOrdersCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "shopify_orders_csv"
    assert unit.source_id.startswith("shopify_orders_csv:")
    assert unit.source_entity_type == "order"
    assert unit.metadata["order_number"] == "#1001"
    assert unit.metadata["created_at"] == "2026-05-01T10:00:00+00:00"
    assert unit.metadata["paid_at"] == "2026-05-01T10:05:00+00:00"
    assert unit.metadata["fulfilled_at"] == "2026-05-02T09:00:00+00:00"
    assert unit.metadata["financial_status"] == "paid"
    assert unit.metadata["fulfillment_status"] == "fulfilled"
    assert unit.metadata["subtotal"] == 40.0
    assert unit.metadata["shipping"] == 5.0
    assert unit.metadata["taxes"] == 3.6
    assert unit.metadata["total"] == 48.6
    assert unit.metadata["lineitem_name"] == "T-Shirt"
    assert unit.metadata["lineitem_quantity"] == 2.0
    assert unit.metadata["customer_email"] == "ada@example.com"
    assert unit.metadata["tags"] == ["vip", "wholesale"]
    assert {"shopify", "order", "paid", "fulfilled", "vip"}.issubset(set(unit.tags))
    assert "Total: 48.6 USD" in unit.content


def test_shopify_orders_csv_aliases_directory_filters_and_deterministic_ids(tmp_path):
    (tmp_path / "old.csv").write_text("Order Date,Order,Item,Total\n2026-05-01,#1,Old,1\n", encoding="utf-8")
    (tmp_path / "new.csv").write_text(
        "Date,Order Number,Line Item Name,Quantity,Price,Email,Payment Status,Fulfilled Status,Order Tags\n"
        "2026-05-03,#2,Mug,1,12.00,grace@example.com,paid,unfulfilled,new\n",
        encoding="utf-8",
    )

    adapter = ShopifyOrdersCsvAdapter(path=str(tmp_path))
    sync = SyncState(source_project="shopify_orders_csv", source_entity_type="order", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.metadata["order_number"] for unit in first.units] == ["#2"]
    assert first.units[0].metadata["lineitem_name"] == "Mug"
    assert first.units[0].metadata["customer_email"] == "grace@example.com"
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["sale"]).units == []


def test_shopify_orders_csv_sparse_rows(tmp_path):
    export = tmp_path / "sparse.csv"
    export.write_text("Name,Lineitem name,Total\n, ,\n#3,Sticker,3.25\n", encoding="utf-8")

    result = ShopifyOrdersCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["order_number"] == "#3"
    assert result.units[0].metadata["total"] == 3.25
