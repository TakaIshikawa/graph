"""Adapter for Capital One transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class CapitalOneTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "capital_one_transactions_csv"

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
        posted_date_text = first(row, "Posted Date", "Post Date")
        transaction_timestamp = parse_datetime(transaction_date_text)
        posted_timestamp = parse_datetime(posted_date_text)
        card_number = first(row, "Card No.", "Card Number", "Card Last Four")
        description = first(row, "Description")
        category = first(row, "Category")
        debit = self._amount(first(row, "Debit"))
        credit = self._amount(first(row, "Credit"))
        explicit_amount = self._amount(first(row, "Amount"))
        balance = self._amount(first(row, "Balance", "Running Balance", "Running Bal."))
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID")
        amount = self._signed_amount(debit, credit, explicit_amount)
        if not any([transaction_date_text, posted_date_text, card_number, description, category, debit is not None, credit is not None, explicit_amount is not None, balance is not None, transaction_id]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_date": self._date(transaction_timestamp, transaction_date_text),
                "posted_date": self._date(posted_timestamp, posted_date_text),
                "card_number": card_number,
                "card_last_four": self._last_four(card_number),
                "description": description,
                "category": category,
                "debit": debit,
                "credit": credit,
                "amount": amount,
                "balance": balance,
                "transaction_id": transaction_id,
                "currency": "USD" if amount is not None else "",
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"capital_one_transactions_csv:{transaction_id}" if transaction_id else digest_source_id(
            "capital_one_transactions_csv",
            transaction_date_text,
            posted_date_text,
            card_number,
            description,
            category,
            debit,
            credit,
            explicit_amount,
            balance,
            index,
        )
        timestamp = transaction_timestamp or posted_timestamp or now
        return KnowledgeUnit(
            source_project="capital_one_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(description, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "capital-one", category] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _signed_amount(self, debit: float | None, credit: float | None, explicit_amount: float | None) -> float | None:
        if debit is not None:
            return -abs(debit)
        if credit is not None:
            return abs(credit)
        return explicit_amount

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

    def _title(self, description: str, amount: float | None) -> str:
        title = description or "Capital One transaction"
        if amount is not None:
            return f"{title} ({amount:g} USD)"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Debit: {metadata.get('debit')}" if metadata.get("debit") is not None else "",
            f"Credit: {metadata.get('credit')}" if metadata.get("credit") is not None else "",
            f"Balance: {metadata.get('balance')}" if metadata.get("balance") is not None else "",
            f"Transaction date: {metadata.get('transaction_date')}" if metadata.get("transaction_date") else "",
            f"Posted date: {metadata.get('posted_date')}" if metadata.get("posted_date") else "",
            f"Card: {metadata.get('card_last_four')}" if metadata.get("card_last_four") else "",
            f"Transaction ID: {metadata.get('transaction_id')}" if metadata.get("transaction_id") else "",
        ]
        return "\n".join(part for part in parts if part)
