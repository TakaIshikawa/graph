"""Adapter for Amazon order history CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import (
    clean_metadata,
    digest_source_id,
    ensure_utc,
    first,
    iter_paths,
    parse_datetime,
    parse_int,
    parse_money,
    read_csv_rows,
)
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class AmazonOrdersCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "amazon_orders_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["order_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or self.entity_types)
        if "order_item" not in allowed:
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

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        order_id = first(row, "Order ID", "Order Id", "order_id")
        order_date_text = first(row, "Order Date", "Purchase Date", "Date")
        shipment_date_text = first(row, "Shipment Date", "Ship Date", "Shipped Date")
        title = first(row, "Title", "Product Name", "Item Name", "Product")
        asin_isbn = first(row, "ASIN/ISBN", "ASIN", "ISBN")
        amount = parse_money(first(row, "Item Total", "Total Owed", "Total", "Amount"))
        if not title:
            return None

        order_date = parse_datetime(order_date_text)
        shipment_date = parse_datetime(shipment_date_text)
        currency = first(row, "Currency", "Currency Code")
        metadata = clean_metadata(
            {
                "order_id": order_id,
                "order_date": order_date.isoformat() if order_date else order_date_text,
                "product_title": title,
                "asin_isbn": asin_isbn,
                "category": first(row, "Category"),
                "seller": first(row, "Seller", "Seller Name"),
                "quantity": parse_int(first(row, "Quantity", "Qty")),
                "amount": amount,
                "currency": currency,
                "shipment_date": shipment_date.isoformat() if shipment_date else shipment_date_text,
                "tracking_number": first(row, "Tracking Number", "Tracking"),
                "ship_to": first(row, "Ship To", "Recipient", "Shipping Address Name"),
                "url": first(row, "URL", "Product URL", "Order URL", "Link"),
                "order_status": first(row, "Order Status", "Status"),
                "source_file": source_file,
            }
        )
        timestamp = shipment_date or order_date
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.AMAZON_ORDERS_CSV,
            source_id=self._source_id(order_id, asin_isbn, title, order_date_text, index),
            source_entity_type="order_item",
            title=title,
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(
                dict.fromkeys(
                    tag
                    for tag in [
                        "amazon",
                        "order_item",
                        metadata.get("category"),
                        metadata.get("seller"),
                        metadata.get("order_status"),
                    ]
                    if tag
                )
            ),
            created_at=order_date or timestamp or now,
            updated_at=timestamp or now,
        )

    def _source_id(self, order_id: str, asin_isbn: str, title: str, order_date: str, index: int) -> str:
        if order_id:
            return digest_source_id("amazon_orders_csv", order_id, asin_isbn, title, index)
        return digest_source_id("amazon_orders_csv", order_date, asin_isbn, title, index)

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            metadata.get("product_title", ""),
            f"Order ID: {metadata.get('order_id')}" if metadata.get("order_id") else "",
            f"Seller: {metadata.get('seller')}" if metadata.get("seller") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip()
            if metadata.get("amount") is not None
            else "",
            f"Status: {metadata.get('order_status')}" if metadata.get("order_status") else "",
            f"Tracking: {metadata.get('tracking_number')}" if metadata.get("tracking_number") else "",
            f"URL: {metadata.get('url')}" if metadata.get("url") else "",
        ]
        return "\n".join(part for part in parts if part)
