"""Adapter for Brex card and cash transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class BrexTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "brex_transactions_csv"

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
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID")
        date_text = first(row, "Date", "Transaction Date", "Timestamp")
        timestamp = parse_datetime(date_text)
        merchant = first(row, "Merchant", "Merchant Name")
        description = first(row, "Description", "Details")
        amount = self._amount(first(row, "Amount"))
        currency = first(row, "Currency")
        status = first(row, "Status", "State")
        category = first(row, "Category")
        card = first(row, "Card", "Card Name", "Card Last Four")
        employee = first(row, "Employee", "User", "Cardholder")
        memo = first(row, "Memo", "Note")
        receipt_url = first(row, "Receipt URL", "Receipt", "Receipt Url")
        if not any([transaction_id, date_text, merchant, description, amount is not None, currency, status, category, card, employee, memo, receipt_url]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "date": timestamp.date().isoformat() if timestamp else date_text,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "merchant": merchant,
                "description": description,
                "amount": amount,
                "currency": currency,
                "status": status,
                "category": category,
                "card": card,
                "employee": employee,
                "memo": memo,
                "receipt_url": receipt_url,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"brex_transactions_csv:{transaction_id}" if transaction_id else digest_source_id(
            "brex_transactions_csv",
            date_text,
            merchant,
            description,
            amount,
            currency,
            card,
            employee,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="brex_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(merchant, description, amount, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "brex", category, status, employee] if tag)),
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

    def _title(self, merchant: str, description: str, amount: float | None, currency: str) -> str:
        title = merchant or description or "Brex transaction"
        if amount is not None:
            suffix = f" {currency}" if currency else ""
            return f"{title} ({amount:g}{suffix})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Merchant: {metadata.get('merchant')}" if metadata.get("merchant") else "",
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Card: {metadata.get('card')}" if metadata.get("card") else "",
            f"Employee: {metadata.get('employee')}" if metadata.get("employee") else "",
            f"Memo: {metadata.get('memo')}" if metadata.get("memo") else "",
            f"Receipt URL: {metadata.get('receipt_url')}" if metadata.get("receipt_url") else "",
            f"Date: {metadata.get('date')}" if metadata.get("date") else "",
            f"Transaction ID: {metadata.get('transaction_id')}" if metadata.get("transaction_id") else "",
        ]
        return "\n".join(part for part in parts if part)
