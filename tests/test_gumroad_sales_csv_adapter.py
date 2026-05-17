from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.gumroad_sales_csv import GumroadSalesCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_gumroad_sales_csv_ingests_sales_metadata_and_status_tags(tmp_path):
    export = tmp_path / "gumroad.csv"
    export.write_text(
        "Sale ID,Order ID,Created At,Product Name,Product ID,Variant,Email,Full Name,Price,Currency,Fee,Net,Quantity,Refunded,Disputed,Status,Affiliate,Discover\n"
        "sale-1,order-1,2026-05-01T08:00:00Z,Guide,prod-1,PDF,ada@example.com,Ada Lovelace,20.00,USD,2.50,17.50,1,false,false,paid,partner@example.com,true\n"
        "sale-2,order-2,2026-05-02T09:00:00Z,Course,prod-2,Video,bob@example.com,Bob,100,USD,10,90,2,true,true,refunded,,false\n",
        encoding="utf-8",
    )

    result = GumroadSalesCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    unit = result.units[0]
    assert unit.source_project == "gumroad_sales_csv"
    assert unit.source_id == "gumroad_sales_csv:sale-1"
    assert unit.source_entity_type == "sale"
    assert unit.content_type == ContentType.METADATA
    assert unit.metadata["sale_id"] == "sale-1"
    assert unit.metadata["order_id"] == "order-1"
    assert unit.metadata["created_at"] == "2026-05-01T08:00:00+00:00"
    assert unit.metadata["product_name"] == "Guide"
    assert unit.metadata["product_id"] == "prod-1"
    assert unit.metadata["variant"] == "PDF"
    assert unit.metadata["email"] == "ada@example.com"
    assert unit.metadata["full_name"] == "Ada Lovelace"
    assert unit.metadata["price"] == 20.0
    assert unit.metadata["fee"] == 2.5
    assert unit.metadata["net"] == 17.5
    assert unit.metadata["quantity"] == 1
    assert unit.metadata["refunded"] is False
    assert unit.metadata["disputed"] is False
    assert unit.metadata["affiliate"] == "partner@example.com"
    assert unit.metadata["discover"] is True
    assert {"gumroad", "sale", "Guide", "paid"}.issubset(set(unit.tags))
    assert {"refunded", "disputed"}.issubset(set(result.units[1].tags))


def test_gumroad_sales_csv_uses_order_id_or_digest_fallback_for_stable_source_ids(tmp_path):
    export = tmp_path / "gumroad.csv"
    export.write_text(
        "Order ID,Created At,Product Name,Email,Price,Currency\n"
        "order-only,2026-05-01,Guide,ada@example.com,20,USD\n"
        ",2026-05-02,Course,bob@example.com,100,USD\n",
        encoding="utf-8",
    )

    first = GumroadSalesCsvAdapter(path=str(export)).ingest().units
    second = GumroadSalesCsvAdapter(path=str(export)).ingest().units

    assert [unit.source_id for unit in first] == [unit.source_id for unit in second]
    assert first[0].source_id == "gumroad_sales_csv:order-only"
    assert first[1].source_id.startswith("gumroad_sales_csv:")


def test_gumroad_sales_csv_skips_blank_rows_sorts_and_filters_since(tmp_path):
    export = tmp_path / "gumroad.csv"
    export.write_text(
        "Sale ID,Created At,Product Name,Price\n"
        ",,,\n"
        "old,2026-05-01,Old,5\n"
        "new,2026-05-03,New,10\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="gumroad_sales_csv", source_entity_type="sale", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = GumroadSalesCsvAdapter(path=str(export)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == ["gumroad_sales_csv:new"]
    assert GumroadSalesCsvAdapter(path=str(export)).ingest(entity_types=["transaction"]).units == []
    all_units = GumroadSalesCsvAdapter(path=str(export)).ingest().units
    assert [(unit.created_at, unit.source_id) for unit in all_units] == sorted((unit.created_at, unit.source_id) for unit in all_units)
