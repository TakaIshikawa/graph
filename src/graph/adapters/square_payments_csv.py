"""Adapter for Square payments CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SquarePaymentsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "square_payments_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["payment"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "payment" not in entity_types:
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
        date_text = first(row, "Date", "Payment Date")
        time_text = first(row, "Time", "Payment Time")
        timestamp_text = first(row, "Date Time", "Date/Time", "Timestamp", "Created At") or " ".join(
            part for part in [date_text, time_text] if part
        )
        timestamp = parse_datetime(timestamp_text)
        timezone_name = first(row, "Timezone", "Time Zone")
        payment_id = first(row, "Payment ID", "Payment Id", "Transaction ID", "Transaction Id")
        order_id = first(row, "Order ID", "Order Id")
        customer_id = first(row, "Customer ID", "Customer Id")
        customer_name = first(row, "Customer Name", "Customer")
        gross_sales = self._money(first(row, "Gross Sales", "Gross Amount"))
        discounts = self._money(first(row, "Discounts", "Discount Amount"))
        net_sales = self._money(first(row, "Net Sales", "Net Amount", "Net Total"))
        tax = self._money(first(row, "Tax", "Taxes"))
        tip = self._money(first(row, "Tip", "Tips"))
        fees = self._money(first(row, "Fees", "Fee", "Processing Fees"))
        total_collected = self._money(first(row, "Total Collected", "Total", "Collected"))
        card_brand = first(row, "Card Brand", "Brand", "Payment Method", "Tender Type")
        last_4 = self._last_four(first(row, "Last 4", "Last Four", "Card Last 4", "Card Last Four"))
        currency = first(row, "Currency", "Currency Code") or self._currency(row) or (
            "USD"
            if any(
                value is not None
                for value in [gross_sales, discounts, net_sales, tax, tip, fees, total_collected]
            )
            else ""
        )

        if not any(
            [
                timestamp_text,
                timezone_name,
                payment_id,
                order_id,
                customer_id,
                customer_name,
                gross_sales is not None,
                discounts is not None,
                net_sales is not None,
                tax is not None,
                tip is not None,
                fees is not None,
                total_collected is not None,
                card_brand,
                last_4,
                currency,
            ]
        ):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "timestamp": timestamp.isoformat() if timestamp else timestamp_text,
                "date": date_text,
                "time": time_text,
                "timezone": timezone_name,
                "gross_sales": gross_sales,
                "discounts": discounts,
                "net_sales": net_sales,
                "tax": tax,
                "tip": tip,
                "fees": fees,
                "total_collected": total_collected,
                "currency": currency,
                "payment_id": payment_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "customer_name": customer_name,
                "card_brand": card_brand,
                "last_4": last_4,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_key = payment_id or order_id
        source_id = (
            f"square_payments_csv:{source_key}"
            if source_key
            else digest_source_id(
                "square_payments_csv",
                timestamp_text,
                customer_id,
                customer_name,
                total_collected,
                card_brand,
                last_4,
                index,
            )
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="square_payments_csv",
            source_id=source_id,
            source_entity_type="payment",
            title=self._title(customer_name, payment_id, total_collected, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["square", "payment", card_brand, customer_name] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _money(self, value: str) -> float | None:
        text = value.strip()
        if not text:
            return None
        negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
        cleaned = re.sub(r"[^0-9.]", "", text)
        if cleaned in {"", "."}:
            return None
        try:
            amount = float(cleaned)
        except ValueError:
            return None
        return -abs(amount) if negative else amount

    def _last_four(self, value: str) -> str:
        digits = re.sub(r"\D", "", value)
        return digits[-4:] if len(digits) >= 4 else ""

    def _currency(self, row: dict[str, Any]) -> str:
        for key in row:
            match = re.search(r"\(([A-Z]{3})\)", str(key))
            if match:
                return match.group(1)
        return ""

    def _title(self, customer_name: str, payment_id: str, total_collected: float | None, currency: str) -> str:
        title = customer_name or payment_id or "Square payment"
        if total_collected is not None:
            suffix = f" {currency}" if currency else ""
            return f"{title} ({total_collected:g}{suffix})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = []
        for key, label in (
            ("timestamp", "Date"),
            ("payment_id", "Payment ID"),
            ("order_id", "Order ID"),
            ("customer_name", "Customer"),
            ("gross_sales", "Gross sales"),
            ("discounts", "Discounts"),
            ("net_sales", "Net sales"),
            ("tax", "Tax"),
            ("tip", "Tip"),
            ("fees", "Fees"),
            ("total_collected", "Total collected"),
            ("card_brand", "Card"),
            ("last_4", "Last 4"),
        ):
            if metadata.get(key) is not None:
                value = metadata[key]
                if key in {"gross_sales", "discounts", "net_sales", "tax", "tip", "fees", "total_collected"} and metadata.get("currency"):
                    value = f"{value} {metadata['currency']}"
                parts.append(f"{label}: {value}")
        return "\n".join(parts)
