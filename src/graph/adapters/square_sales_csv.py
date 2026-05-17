"""Adapter for Square sales CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SquareSalesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "square_sales_csv"

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
        transaction_id = first(row, "Transaction ID", "Transaction Id", "Payment ID", "Payment Id", "Order ID")
        receipt_number = first(row, "Receipt Number", "Receipt #", "Receipt")
        date_text = first(row, "Date", "Date Time", "Date/Time", "Transaction Time", "Payment Time")
        timestamp = parse_datetime(date_text)
        location = first(row, "Location", "Location Name")
        customer = first(row, "Customer", "Customer Name")
        item = first(row, "Item", "Item Name", "Description")
        variation = first(row, "Variation", "Variation Name")
        category = first(row, "Category")
        sku = first(row, "SKU", "Sku")
        quantity = parse_float(first(row, "Quantity", "Qty"))
        gross_sales = parse_float(first(row, "Gross Sales", "Gross Amount"))
        discounts = parse_float(first(row, "Discounts", "Discount Amount"))
        tax = parse_float(first(row, "Tax", "Taxes"))
        tip = parse_float(first(row, "Tip", "Tips"))
        net_total = parse_float(first(row, "Net Total", "Net Sales", "Total"))
        payment_method = first(row, "Payment Method", "Tender Type", "Card Brand")
        currency = first(row, "Currency", "Currency Code")

        if not any([transaction_id, receipt_number, date_text, location, customer, item, variation, category, sku, quantity is not None, gross_sales is not None, discounts is not None, tax is not None, tip is not None, net_total is not None, payment_method, currency]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "receipt_number": receipt_number,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "location": location,
                "customer": customer,
                "item": item,
                "variation": variation,
                "category": category,
                "sku": sku,
                "quantity": quantity,
                "gross_sales": gross_sales,
                "discounts": discounts,
                "tax": tax,
                "tip": tip,
                "net_total": net_total,
                "payment_method": payment_method,
                "currency": currency,
                "source_file": source_file,
            }
        )
        source_key = transaction_id or receipt_number
        return KnowledgeUnit(
            source_project="square_sales_csv",
            source_id=f"square_sales_csv:{source_key}" if source_key else digest_source_id("square_sales_csv", date_text, location, item, sku, net_total, index),
            source_entity_type="sale",
            title=self._title(item, receipt_number, net_total, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["square", "sale", location, category, payment_method] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _title(self, item: str, receipt_number: str, net_total: float | None, currency: str) -> str:
        title = item or receipt_number or "Square sale"
        if net_total is not None:
            return f"{title} ({net_total:g} {currency})".strip()
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = []
        for key, label in (
            ("timestamp", "Date"),
            ("receipt_number", "Receipt"),
            ("location", "Location"),
            ("customer", "Customer"),
            ("item", "Item"),
            ("quantity", "Quantity"),
            ("gross_sales", "Gross sales"),
            ("discounts", "Discounts"),
            ("tax", "Tax"),
            ("tip", "Tip"),
            ("net_total", "Net total"),
            ("payment_method", "Payment"),
        ):
            if metadata.get(key) is not None:
                value = metadata[key]
                if key in {"gross_sales", "discounts", "tax", "tip", "net_total"} and metadata.get("currency"):
                    value = f"{value} {metadata['currency']}"
                parts.append(f"{label}: {value}")
        return "\n".join(parts)
