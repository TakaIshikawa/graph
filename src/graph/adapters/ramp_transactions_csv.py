"""Adapter for Ramp corporate card transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class RampTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "ramp_transactions_csv"

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
        transaction_id = first(row, "Transaction ID", "ID")
        date_text = first(row, "Date", "Transaction Date")
        timestamp = parse_datetime(date_text)
        merchant = first(row, "Merchant", "Merchant Name")
        amount = self._amount(first(row, "Amount"))
        currency = first(row, "Currency") or ("USD" if amount is not None else "")
        user = first(row, "User", "Cardholder")
        department = first(row, "Department")
        memo = first(row, "Memo", "Note")
        category = first(row, "Category")
        receipt = first(row, "Receipt", "Receipt URL")
        card_last_four = first(row, "Card Last Four", "Card Last 4", "Card")
        status = first(row, "State/Status", "State", "Status")
        if not any([transaction_id, date_text, merchant, amount is not None, currency, user, department, memo, category, receipt, card_last_four, status]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "date": self._date(timestamp, date_text),
                "merchant": merchant,
                "amount": amount,
                "currency": currency,
                "user": user,
                "department": department,
                "memo": memo,
                "category": category,
                "receipt": receipt,
                "card_last_four": self._last_four(card_last_four) or card_last_four,
                "status": status,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"ramp_transactions_csv:{transaction_id}" if transaction_id else digest_source_id(
            "ramp_transactions_csv",
            date_text,
            merchant,
            amount,
            currency,
            user,
            department,
            card_last_four,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="ramp_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(merchant, amount, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "ramp", category, department, status] if tag)),
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

    def _last_four(self, value: str) -> str:
        digits = re.sub(r"\D", "", value)
        return digits[-4:] if len(digits) >= 4 else ""

    def _date(self, timestamp: datetime | None, fallback: str) -> str:
        return timestamp.date().isoformat() if timestamp else fallback

    def _title(self, merchant: str, amount: float | None, currency: str) -> str:
        title = merchant or "Ramp transaction"
        if amount is not None:
            suffix = f" {currency}" if currency else ""
            return f"{title} ({amount:g}{suffix})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Merchant: {metadata.get('merchant')}" if metadata.get("merchant") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Date: {metadata.get('date')}" if metadata.get("date") else "",
            f"User: {metadata.get('user')}" if metadata.get("user") else "",
            f"Department: {metadata.get('department')}" if metadata.get("department") else "",
            f"Memo: {metadata.get('memo')}" if metadata.get("memo") else "",
            f"Receipt: {metadata.get('receipt')}" if metadata.get("receipt") else "",
        ]
        return "\n".join(part for part in parts if part)
