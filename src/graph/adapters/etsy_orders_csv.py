"""Adapter for Etsy orders CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class EtsyOrdersCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "etsy_orders_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["order"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "order" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        order_id = first(row, "Order ID", "Order Id", "Receipt ID", "Receipt Id")
        sale_date_text = first(row, "Sale Date", "Date", "Order Date")
        sale_date = parse_datetime(sale_date_text)
        buyer = first(row, "Buyer", "Buyer Name", "Customer")
        shop_name = first(row, "Shop Name", "Shop")
        item_title = first(row, "Item Title", "Title", "Item")
        listing_id = first(row, "Listing ID", "Listing Id")
        sku = first(row, "SKU", "Sku")
        quantity = parse_float(first(row, "Quantity", "Qty"))
        price = parse_float(first(row, "Price", "Item Price"))
        coupon_discount = parse_float(first(row, "Coupon Discount", "Discount", "Discount Amount"))
        shipping = parse_float(first(row, "Shipping", "Shipping Total"))
        tax = parse_float(first(row, "Tax", "Sales Tax"))
        order_total = parse_float(first(row, "Order Total", "Total", "Grand Total"))
        currency = first(row, "Currency", "Currency Code")
        ship_date_text = first(row, "Ship Date", "Shipped Date", "Date Shipped")
        ship_date = parse_datetime(ship_date_text)
        destination = first(row, "Destination", "Ship To", "Shipping Address", "Country")
        order_status = first(row, "Order Status", "Status")

        if not any([order_id, sale_date_text, buyer, shop_name, item_title, listing_id, sku, quantity is not None, price is not None, coupon_discount is not None, shipping is not None, tax is not None, order_total is not None, currency, ship_date_text, destination, order_status]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "order_id": order_id,
                "sale_date": sale_date.isoformat() if sale_date else sale_date_text,
                "buyer": buyer,
                "shop_name": shop_name,
                "item_title": item_title,
                "listing_id": listing_id,
                "sku": sku,
                "quantity": quantity,
                "price": price,
                "coupon_discount": coupon_discount,
                "shipping": shipping,
                "tax": tax,
                "order_total": order_total,
                "currency": currency,
                "ship_date": ship_date.isoformat() if ship_date else ship_date_text,
                "destination": destination,
                "order_status": order_status,
                "source_file": source_file,
            }
        )
        source_key = ":".join(part for part in [order_id, listing_id] if part)
        return KnowledgeUnit(
            source_project="etsy_orders_csv",
            source_id=f"etsy_orders_csv:{source_key}" if source_key else digest_source_id("etsy_orders_csv", sale_date_text, buyer, item_title, sku, order_total, index),
            source_entity_type="order",
            title=self._title(order_id, item_title, buyer, order_total, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["etsy", "order", shop_name, order_status, destination] if tag)),
            created_at=sale_date or ship_date or now,
            updated_at=ship_date or sale_date or now,
        )

    def _title(self, order_id: str, item_title: str, buyer: str, order_total: float | None, currency: str) -> str:
        title = " - ".join(part for part in [order_id, item_title, buyer] if part) or "Etsy order"
        if order_total is not None:
            return f"{title} ({order_total:g} {currency})".strip()
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = []
        for key, label in (
            ("sale_date", "Sale date"),
            ("buyer", "Buyer"),
            ("shop_name", "Shop"),
            ("item_title", "Item"),
            ("quantity", "Quantity"),
            ("price", "Price"),
            ("coupon_discount", "Coupon discount"),
            ("shipping", "Shipping"),
            ("tax", "Tax"),
            ("order_total", "Order total"),
            ("ship_date", "Ship date"),
            ("destination", "Destination"),
            ("order_status", "Status"),
        ):
            if metadata.get(key) is not None:
                value = metadata[key]
                if key in {"price", "coupon_discount", "shipping", "tax", "order_total"} and metadata.get("currency"):
                    value = f"{value} {metadata['currency']}"
                parts.append(f"{label}: {value}")
        return "\n".join(parts)
