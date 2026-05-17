"""Adapter for Google Pay and Wallet transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GooglePayTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_pay_transactions_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["transaction"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "transaction" not in entity_types:
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
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID", "Google Transaction ID")
        date_text = first(row, "Transaction Date", "Date", "Timestamp", "Time", "Created")
        timestamp = self._timestamp(date_text)
        merchant = first(row, "Merchant", "Payee", "Business", "Store", "Name")
        description = first(row, "Description", "Item", "Details", "Memo")
        amount = self._amount(first(row, "Amount", "Total", "Transaction Amount"))
        currency = first(row, "Currency", "Currency Code")
        status = first(row, "Status", "Transaction Status")
        payment_method = first(row, "Payment Method", "Funding Source", "Card", "Instrument")
        category = first(row, "Category", "Type")
        notes = first(row, "Notes", "Note", "Comment")
        if not any([transaction_id, date_text, merchant, description, amount is not None, status, payment_method, category, notes]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "merchant": merchant,
                "description": description,
                "amount": amount,
                "currency": currency,
                "status": status,
                "payment_method": payment_method,
                "category": category,
                "notes": notes,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="google_pay_transactions_csv",
            source_id=f"google_pay_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("google_pay_transactions_csv", date_text, merchant, description, amount, currency, index),
            source_entity_type="transaction",
            title=self._title(merchant, description, amount, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["google_pay", "transaction", status, category, currency] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _timestamp(self, value: str) -> datetime | None:
        parsed = parse_datetime(value)
        if parsed:
            return parsed
        text = value.strip()
        if not text:
            return None
        text = re.sub(r"\s+(UTC|GMT)$", "", text, flags=re.IGNORECASE)
        for fmt in (
            "%b %d, %Y, %I:%M:%S %p",
            "%B %d, %Y, %I:%M:%S %p",
            "%b %d, %Y, %I:%M %p",
            "%B %d, %Y, %I:%M %p",
            "%m/%d/%Y %I:%M:%S %p",
            "%m/%d/%Y %I:%M %p",
        ):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _amount(self, value: str) -> float | None:
        text = value.strip()
        if not text:
            return None
        negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
        cleaned = re.sub(r"[^0-9.]", "", text)
        if not cleaned:
            return None
        try:
            amount = float(cleaned)
        except ValueError:
            return None
        return -abs(amount) if negative else amount

    def _title(self, merchant: str, description: str, amount: float | None, currency: str) -> str:
        title = merchant or description or "Google Pay transaction"
        if amount is not None:
            return f"{title} ({amount:g} {currency})".strip()
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Merchant: {metadata.get('merchant')}" if metadata.get("merchant") else "",
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Payment method: {metadata.get('payment_method')}" if metadata.get("payment_method") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Notes: {metadata.get('notes')}" if metadata.get("notes") else "",
        ]
        return "\n".join(part for part in parts if part)
