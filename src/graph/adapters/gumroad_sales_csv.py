"""Adapter for Gumroad sales CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GumroadSalesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "gumroad_sales_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["sale"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "sale" not in entity_types:
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
        sale_id = first(row, "Sale ID", "Sale Id", "ID", "Id")
        order_id = first(row, "Order ID", "Order Id", "Order Number")
        created_text = first(row, "Created At", "Created", "Sale Date", "Date")
        created_at = parse_datetime(created_text)
        product_name = first(row, "Product Name", "Product")
        product_id = first(row, "Product ID", "Product Id")
        variant = first(row, "Variant", "Variant Name")
        email = first(row, "Email", "Customer Email")
        full_name = first(row, "Full Name", "Customer Name", "Name")
        price = parse_float(first(row, "Price", "Sale Price", "Amount"))
        currency = first(row, "Currency")
        fee = parse_float(first(row, "Fee", "Gumroad Fee"))
        net = parse_float(first(row, "Net", "Net Amount", "Net Revenue"))
        quantity = parse_int(first(row, "Quantity", "Qty"))
        refunded = self._bool(first(row, "Refunded", "Is Refunded", "Refund Status"))
        disputed = self._bool(first(row, "Disputed", "Is Disputed", "Chargeback", "Chargebacked"))
        status = first(row, "Status", "Payment Status")
        affiliate = first(row, "Affiliate", "Affiliate Email")
        discover = self._bool(first(row, "Discover", "Gumroad Discover"))

        if not any([sale_id, order_id, created_text, product_name, product_id, variant, email, full_name, price is not None, fee is not None, net is not None, quantity is not None, refunded is not None, disputed is not None, status]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "sale_id": sale_id,
                "order_id": order_id,
                "created_at": created_at.isoformat() if created_at else created_text,
                "product_name": product_name,
                "product_id": product_id,
                "variant": variant,
                "email": email,
                "full_name": full_name,
                "price": price,
                "currency": currency,
                "fee": fee,
                "net": net,
                "quantity": quantity,
                "refunded": refunded,
                "disputed": disputed,
                "status": status,
                "affiliate": affiliate,
                "discover": discover,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        timestamp = created_at or now
        return KnowledgeUnit(
            source_project="gumroad_sales_csv",
            source_id=self._source_id(sale_id, order_id, created_text, product_name, email, price, index),
            source_entity_type="sale",
            title=self._title(product_name, full_name, email, price, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=self._tags(product_name, status, refunded, disputed),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _source_id(self, sale_id: str, order_id: str, created_text: str, product_name: str, email: str, price: float | None, index: int) -> str:
        if sale_id:
            return f"gumroad_sales_csv:{sale_id}"
        if order_id:
            return f"gumroad_sales_csv:{order_id}"
        return digest_source_id("gumroad_sales_csv", created_text, product_name, email, price, index)

    def _bool(self, value: str) -> bool | None:
        text = value.strip().casefold()
        if not text:
            return None
        if text in {"true", "yes", "y", "1", "refunded", "disputed", "chargeback", "chargebacked"}:
            return True
        if text in {"false", "no", "n", "0", "not refunded", "none", "succeeded", "paid"}:
            return False
        return None

    def _tags(self, product_name: str, status: str, refunded: bool | None, disputed: bool | None) -> list[str]:
        tags = ["gumroad", "sale", product_name, status]
        if refunded:
            tags.append("refunded")
        if disputed:
            tags.append("disputed")
        return list(dict.fromkeys(tag for tag in tags if tag))

    def _title(self, product_name: str, full_name: str, email: str, price: float | None, currency: str) -> str:
        title = product_name or "Gumroad sale"
        customer = full_name or email
        if customer:
            title = f"{title} - {customer}"
        if price is not None:
            return f"{title} ({price:g} {currency})".strip()
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Product: {metadata.get('product_name')}" if metadata.get("product_name") else "",
            f"Customer: {metadata.get('full_name') or metadata.get('email')}" if metadata.get("full_name") or metadata.get("email") else "",
            f"Price: {metadata.get('price')} {metadata.get('currency', '')}".strip() if metadata.get("price") is not None else "",
            f"Fee: {metadata.get('fee')}" if metadata.get("fee") is not None else "",
            f"Net: {metadata.get('net')}" if metadata.get("net") is not None else "",
            f"Quantity: {metadata.get('quantity')}" if metadata.get("quantity") is not None else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
        ]
        return "\n".join(part for part in parts if part)
