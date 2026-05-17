"""Adapter for Shopify orders CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class ShopifyOrdersCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "shopify_orders_csv"

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
        order_number = first(row, "Name", "Order Number", "Order", "Order Name")
        created_text = first(row, "Created at", "Created At", "Created", "Order Date", "Date")
        paid_text = first(row, "Paid at", "Paid At", "Paid Date")
        fulfilled_text = first(row, "Fulfilled at", "Fulfilled At", "Fulfillment Date")
        created_at = parse_datetime(created_text)
        paid_at = parse_datetime(paid_text)
        fulfilled_at = parse_datetime(fulfilled_text)
        financial_status = first(row, "Financial Status", "Payment Status")
        fulfillment_status = first(row, "Fulfillment Status", "Fulfilled Status")
        currency = first(row, "Currency")
        subtotal = parse_float(first(row, "Subtotal"))
        shipping = parse_float(first(row, "Shipping", "Shipping Total"))
        taxes = parse_float(first(row, "Taxes", "Tax"))
        total = parse_float(first(row, "Total", "Order Total"))
        discount_amount = parse_float(first(row, "Discount Amount", "Discount"))
        lineitem_name = first(row, "Lineitem name", "Lineitem Name", "Line Item Name", "Item")
        lineitem_quantity = parse_float(first(row, "Lineitem quantity", "Lineitem Quantity", "Quantity"))
        lineitem_price = parse_float(first(row, "Lineitem price", "Lineitem Price", "Price")
        )
        sku = first(row, "SKU", "Lineitem sku", "Lineitem SKU")
        customer_name = first(row, "Customer name", "Customer Name", "Customer")
        customer_email = first(row, "Customer email", "Customer Email", "Email")
        billing_country = first(row, "Billing country", "Billing Country")
        shipping_country = first(row, "Shipping country", "Shipping Country")
        tags = split_values(first(row, "Tags", "Order Tags"))
        note = first(row, "Note", "Notes")

        if not any([order_number, created_text, paid_text, fulfilled_text, financial_status, fulfillment_status, currency, subtotal is not None, shipping is not None, taxes is not None, total is not None, discount_amount is not None, lineitem_name, lineitem_quantity is not None, lineitem_price is not None, sku, customer_name, customer_email, billing_country, shipping_country, tags, note]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "order_number": order_number,
                "created_at": created_at.isoformat() if created_at else created_text,
                "paid_at": paid_at.isoformat() if paid_at else paid_text,
                "fulfilled_at": fulfilled_at.isoformat() if fulfilled_at else fulfilled_text,
                "financial_status": financial_status,
                "fulfillment_status": fulfillment_status,
                "currency": currency,
                "subtotal": subtotal,
                "shipping": shipping,
                "taxes": taxes,
                "total": total,
                "discount_amount": discount_amount,
                "lineitem_name": lineitem_name,
                "lineitem_quantity": lineitem_quantity,
                "lineitem_price": lineitem_price,
                "sku": sku,
                "customer_name": customer_name,
                "customer_email": customer_email,
                "billing_country": billing_country,
                "shipping_country": shipping_country,
                "tags": tags,
                "note": note,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="shopify_orders_csv",
            source_id=self._source_id(order_number, lineitem_name, sku, index, created_text),
            source_entity_type="order",
            title=self._title(order_number, lineitem_name, total, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["shopify", "order", financial_status, fulfillment_status, *tags] if tag)),
            created_at=created_at or paid_at or fulfilled_at or now,
            updated_at=fulfilled_at or paid_at or created_at or now,
        )

    def _source_id(self, order_number: str, lineitem_name: str, sku: str, index: int, created_text: str) -> str:
        if order_number:
            parts = [order_number]
            if lineitem_name or sku:
                parts.extend([lineitem_name, sku])
            return "shopify_orders_csv:" + digest_source_id("", *parts).split(":", 1)[1]
        return digest_source_id("shopify_orders_csv", created_text, lineitem_name, sku, index)

    def _title(self, order_number: str, lineitem_name: str, total: float | None, currency: str) -> str:
        title = " - ".join(part for part in [order_number, lineitem_name] if part) or "Shopify order"
        if total is not None:
            return f"{title} ({total:g} {currency})".strip()
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = []
        for key, label in (
            ("created_at", "Created"),
            ("paid_at", "Paid"),
            ("fulfilled_at", "Fulfilled"),
            ("financial_status", "Financial status"),
            ("fulfillment_status", "Fulfillment status"),
            ("lineitem_name", "Line item"),
            ("lineitem_quantity", "Quantity"),
            ("subtotal", "Subtotal"),
            ("shipping", "Shipping"),
            ("taxes", "Taxes"),
            ("discount_amount", "Discount"),
            ("total", "Total"),
            ("customer_name", "Customer"),
            ("customer_email", "Email"),
            ("billing_country", "Billing country"),
            ("shipping_country", "Shipping country"),
            ("note", "Note"),
        ):
            if metadata.get(key) is not None:
                value = metadata[key]
                if key in {"subtotal", "shipping", "taxes", "discount_amount", "total", "lineitem_price"} and metadata.get("currency"):
                    value = f"{value} {metadata['currency']}"
                parts.append(f"{label}: {value}")
        return "\n".join(parts)
