"""Adapter for Apple Card transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AppleCardTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "apple_card_transactions_csv"

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
        transaction_date_text = first(row, "Transaction Date", "Date")
        clearing_date_text = first(row, "Clearing Date", "Posting Date", "Posted Date")
        transaction_timestamp = parse_datetime(transaction_date_text)
        clearing_timestamp = parse_datetime(clearing_date_text)
        description = first(row, "Description")
        merchant = first(row, "Merchant", "Merchant Name")
        category = first(row, "Category")
        transaction_type = first(row, "Type", "Transaction Type")
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID")
        amount = self._amount(first(row, "Amount (USD)", "Amount", "Amount USD"))
        location = first(row, "Location", "Merchant Location", "Address")
        purchased_by = first(row, "Purchased By", "User")
        last_four_digits = first(row, "Last Four Digits", "Last 4 Digits", "Card Last Four")
        if not any([transaction_id, transaction_date_text, clearing_date_text, description, merchant, category, transaction_type, amount is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "transaction_date": self._date(transaction_timestamp, transaction_date_text),
                "clearing_date": self._date(clearing_timestamp, clearing_date_text),
                "description": description,
                "merchant": merchant,
                "category": category,
                "type": transaction_type,
                "amount": amount,
                "currency": "USD" if amount is not None else "",
                "location": location,
                "purchased_by": purchased_by,
                "last_four_digits": last_four_digits,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = (
            f"apple_card_transactions_csv:{transaction_id}"
            if transaction_id
            else digest_source_id(
                "apple_card_transactions_csv",
                transaction_date_text,
                clearing_date_text,
                description,
                merchant,
                transaction_type,
                amount,
                location,
                purchased_by,
                last_four_digits,
                index,
            )
        )
        timestamp = transaction_timestamp or clearing_timestamp or now
        return KnowledgeUnit(
            source_project="apple_card_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(merchant, description, transaction_type, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "apple-card", category, transaction_type] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _amount(self, value: str) -> float | None:
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

    def _date(self, timestamp: datetime | None, fallback: str) -> str:
        return timestamp.date().isoformat() if timestamp else fallback

    def _title(self, merchant: str, description: str, transaction_type: str, amount: float | None) -> str:
        title = merchant or description or transaction_type or "Apple Card transaction"
        if amount is not None:
            return f"{title} ({amount:g} USD)"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Merchant: {metadata.get('merchant')}" if metadata.get("merchant") else "",
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Transaction date: {metadata.get('transaction_date')}" if metadata.get("transaction_date") else "",
            f"Clearing date: {metadata.get('clearing_date')}" if metadata.get("clearing_date") else "",
        ]
        return "\n".join(part for part in parts if part)
