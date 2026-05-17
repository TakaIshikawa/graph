from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.etsy_orders_csv import EtsyOrdersCsvAdapter
from graph.types.models import SyncState


def test_etsy_orders_csv_ingests_order_metadata(tmp_path):
    export = tmp_path / "etsy.csv"
    export.write_text(
        "Order ID,Sale Date,Buyer,Shop Name,Item Title,Listing ID,SKU,Quantity,Price,Coupon Discount,Shipping,Tax,Order Total,Currency,Ship Date,Destination,Order Status\n"
        "E100,2026-05-01,Ada,My Shop,Handmade Mug,L200,MUG-1,2,20.00,5.00,7.50,1.80,44.30,USD,2026-05-03,US,Shipped\n",
        encoding="utf-8",
    )

    unit = EtsyOrdersCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "etsy_orders_csv"
    assert unit.source_id == "etsy_orders_csv:E100:L200"
    assert unit.source_entity_type == "order"
    assert unit.metadata["sale_date"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["buyer"] == "Ada"
    assert unit.metadata["quantity"] == 2.0
    assert unit.metadata["order_total"] == 44.3
    assert unit.metadata["ship_date"] == "2026-05-03T00:00:00+00:00"
    assert unit.metadata["destination"] == "US"
    assert unit.metadata["order_status"] == "Shipped"
    assert {"etsy", "order", "My Shop", "Shipped", "US"}.issubset(set(unit.tags))
    assert "Order total: 44.3 USD" in unit.content


def test_etsy_orders_csv_aliases_directory_filters_and_deterministic_order(tmp_path):
    (tmp_path / "b.csv").write_text("Order Date,Receipt ID,Title,Total\n2026-05-03,E300,New,3\n", encoding="utf-8")
    (tmp_path / "a.csv").write_text("Date,Order Id,Item,Grand Total\n2026-05-01,E100,Old,1\n", encoding="utf-8")

    adapter = EtsyOrdersCsvAdapter(path=str(tmp_path))
    sync = SyncState(source_project="etsy_orders_csv", source_entity_type="order", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.metadata["item_title"] for unit in first.units] == ["New"]
    assert [unit.source_id for unit in first.units] == ["etsy_orders_csv:E300"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert [unit.source_id for unit in adapter.ingest().units] == ["etsy_orders_csv:E100", "etsy_orders_csv:E300"]
    assert adapter.ingest(entity_types=["sale"]).units == []


def test_etsy_orders_csv_sparse_rows(tmp_path):
    export = tmp_path / "sparse.csv"
    export.write_text("Buyer,Item Title,Order Total\n, ,\nGrace,Sticker,3.25\n", encoding="utf-8")

    result = EtsyOrdersCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["buyer"] == "Grace"
    assert result.units[0].source_id.startswith("etsy_orders_csv:")
