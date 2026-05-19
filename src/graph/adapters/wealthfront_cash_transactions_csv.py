"""Adapter for Wealthfront Cash Account transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class WealthfrontCashTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "wealthfront_cash_transactions_csv"

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
        date_text = first(row, "Date", "Transaction Date", "Posted Date")
        timestamp = parse_datetime(date_text)
        description = first(row, "Description", "Merchant", "Name")
        category = first(row, "Category")
        amount = self._amount(first(row, "Amount", "Transaction Amount"))
        balance = self._amount(first(row, "Balance", "Running Balance"))
        account = first(row, "Account", "Account Name")
        institution = first(row, "Institution", "Bank")
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID")
        if not any([date_text, description, category, amount is not None, balance is not None, account, institution, transaction_id]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "date": self._date(timestamp, date_text),
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "description": description,
                "category": category,
                "amount": amount,
                "currency": "USD" if amount is not None else "",
                "balance": balance,
                "account": account,
                "institution": institution,
                "transaction_id": transaction_id,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"wealthfront_cash_transactions_csv:{transaction_id}" if transaction_id else digest_source_id(
            "wealthfront_cash_transactions_csv",
            date_text,
            description,
            category,
            amount,
            balance,
            account,
            institution,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="wealthfront_cash_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(description, category, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "wealthfront", "cash", category] if tag)),
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

    def _title(self, description: str, category: str, amount: float | None) -> str:
        title = description or category or "Wealthfront cash transaction"
        if amount is not None:
            return f"{title} ({amount:g} USD)"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Balance: {metadata.get('balance')}" if metadata.get("balance") is not None else "",
            f"Date: {metadata.get('date')}" if metadata.get("date") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Institution: {metadata.get('institution')}" if metadata.get("institution") else "",
            f"Transaction ID: {metadata.get('transaction_id')}" if metadata.get("transaction_id") else "",
        ]
        return "\n".join(part for part in parts if part)
