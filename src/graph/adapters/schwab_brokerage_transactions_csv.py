"""Adapter for Schwab brokerage transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SchwabBrokerageTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "schwab_brokerage_transactions_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["brokerage_transaction"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "brokerage_transaction" not in entity_types:
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
        date_text = first(row, "Date", "Transaction Date")
        timestamp = parse_datetime(date_text)
        action = first(row, "Action", "Type")
        symbol = first(row, "Symbol")
        description = first(row, "Description")
        quantity = self._amount(first(row, "Quantity", "Qty"))
        price = self._amount(first(row, "Price"))
        fees_commission = self._amount(first(row, "Fees & Comm", "Fees and Comm", "Fees", "Commission"))
        amount = self._amount(first(row, "Amount"))
        account = first(row, "Account", "Account Number")
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID")
        if not any([transaction_id, date_text, action, symbol, description, quantity is not None, price is not None, fees_commission is not None, amount is not None, account]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "date": timestamp.date().isoformat() if timestamp else date_text,
                "action": action,
                "symbol": symbol,
                "description": description,
                "quantity": quantity,
                "price": price,
                "fees_commission": fees_commission,
                "amount": amount,
                "account": account,
                "transaction_id": transaction_id,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="schwab_brokerage_transactions_csv",
            source_id=f"schwab_brokerage_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("schwab_brokerage_transactions_csv", date_text, action, symbol, description, amount, account, index),
            source_entity_type="brokerage_transaction",
            title=self._title(action, symbol, description, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "schwab", "brokerage_transaction", action, symbol] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
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

    def _title(self, action: str, symbol: str, description: str, amount: float | None) -> str:
        title = " ".join(part for part in [action, symbol] if part) or description or "Schwab brokerage transaction"
        if amount is not None:
            return f"{title} ({amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Action: {metadata.get('action')}" if metadata.get("action") else "",
            f"Symbol: {metadata.get('symbol')}" if metadata.get("symbol") else "",
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Quantity: {metadata.get('quantity')}" if metadata.get("quantity") is not None else "",
            f"Price: {metadata.get('price')}" if metadata.get("price") is not None else "",
            f"Fees/Commission: {metadata.get('fees_commission')}" if metadata.get("fees_commission") is not None else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Date: {metadata.get('date')}" if metadata.get("date") else "",
            f"Transaction ID: {metadata.get('transaction_id')}" if metadata.get("transaction_id") else "",
        ]
        return "\n".join(part for part in parts if part)
