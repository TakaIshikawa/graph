"""Adapter for Chase credit card transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class ChaseCreditCardTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "chase_credit_card_transactions_csv"

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
        transaction_date_text = first(row, "Transaction Date", "Date")
        post_date_text = first(row, "Post Date", "Posting Date", "Posted Date")
        transaction_timestamp = parse_datetime(transaction_date_text)
        post_timestamp = parse_datetime(post_date_text)
        description = first(row, "Description", "Merchant")
        category = first(row, "Category")
        transaction_type = first(row, "Type", "Transaction Type")
        raw_amount = first(row, "Amount")
        amount = self._signed_amount(raw_amount, transaction_type)
        memo = first(row, "Memo", "Notes")
        account = first(row, "Account", "Account Name")
        card = first(row, "Card", "Card Number", "Card No.")
        if not any([transaction_id, transaction_date_text, post_date_text, description, category, transaction_type, amount is not None, memo, account, card]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_date": self._date(transaction_timestamp, transaction_date_text),
                "post_date": self._date(post_timestamp, post_date_text),
                "description": description,
                "category": category,
                "type": transaction_type,
                "amount": amount,
                "currency": "USD" if amount is not None else "",
                "memo": memo,
                "account": account,
                "card": card,
                "transaction_id": transaction_id,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        timestamp = transaction_timestamp or post_timestamp or now
        return KnowledgeUnit(
            source_project="chase_credit_card_transactions_csv",
            source_id=f"chase_credit_card_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("chase_credit_card_transactions_csv", transaction_date_text, post_date_text, description, category, transaction_type, raw_amount, memo, account, card, index),
            source_entity_type="transaction",
            title=self._title(description, transaction_type, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "chase", "credit-card", category, transaction_type] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _signed_amount(self, value: str, transaction_type: str) -> float | None:
        amount = self._amount(value)
        if amount is None:
            return None
        normalized_type = transaction_type.casefold()
        if any(word in normalized_type for word in ("payment", "credit", "return", "refund")):
            return abs(amount)
        if any(word in normalized_type for word in ("sale", "debit", "purchase", "charge")):
            return -abs(amount)
        return amount

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

    def _title(self, description: str, transaction_type: str, amount: float | None) -> str:
        title = description or transaction_type or "Chase credit card transaction"
        if amount is not None:
            return f"{title} ({amount:g} USD)"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Transaction date: {metadata.get('transaction_date')}" if metadata.get("transaction_date") else "",
            f"Post date: {metadata.get('post_date')}" if metadata.get("post_date") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Card: {metadata.get('card')}" if metadata.get("card") else "",
            f"Memo: {metadata.get('memo')}" if metadata.get("memo") else "",
            f"Transaction ID: {metadata.get('transaction_id')}" if metadata.get("transaction_id") else "",
        ]
        return "\n".join(part for part in parts if part)
