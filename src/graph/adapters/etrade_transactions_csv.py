"""Adapter for ETRADE brokerage transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class EtradeTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "etrade_transactions_csv"

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
        date_text = first(row, "TransactionDate", "Transaction Date", "Date")
        timestamp = parse_datetime(date_text)
        transaction_type = first(row, "Transaction Type", "TransactionType", "Type")
        security_type = first(row, "Security Type", "SecurityType")
        symbol = first(row, "Symbol")
        quantity = parse_money(first(row, "Quantity", "Qty"))
        amount = parse_money(first(row, "Amount"))
        description = first(row, "Description")
        transaction_id = first(row, "Transaction ID", "TransactionId", "ID")
        if not any([transaction_id, date_text, transaction_type, security_type, symbol, quantity is not None, amount is not None, description]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "transaction_type": transaction_type,
                "security_type": security_type,
                "symbol": symbol,
                "quantity": quantity,
                "amount": amount,
                "description": description,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.ETRADE_TRANSACTIONS_CSV,
            source_id=f"etrade_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("etrade_transactions_csv", date_text, transaction_type, security_type, symbol, quantity, amount, index),
            source_entity_type="brokerage_transaction",
            title=self._title(transaction_type, symbol, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["etrade", "brokerage_transaction", transaction_type, security_type, symbol] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _title(self, transaction_type: str, symbol: str, amount: float | None) -> str:
        title = " ".join(part for part in [transaction_type, symbol] if part) or "ETRADE transaction"
        if amount is not None:
            return f"{title} ({amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Type: {metadata.get('transaction_type')}" if metadata.get("transaction_type") else "",
            f"Security type: {metadata.get('security_type')}" if metadata.get("security_type") else "",
            f"Symbol: {metadata.get('symbol')}" if metadata.get("symbol") else "",
            metadata.get("description", ""),
            f"Quantity: {metadata.get('quantity')}" if metadata.get("quantity") is not None else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
        ]
        return "\n".join(part for part in parts if part)
