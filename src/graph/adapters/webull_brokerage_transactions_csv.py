"""Adapter for Webull brokerage transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class WebullBrokerageTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "webull_brokerage_transactions_csv"

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
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID")
        trade_date_text = first(row, "Trade Date", "Date", "Transaction Date")
        timestamp = parse_datetime(trade_date_text)
        settlement_date_text = first(row, "Settlement Date", "Settle Date")
        settlement_date = parse_datetime(settlement_date_text)
        symbol = first(row, "Symbol", "Ticker")
        name = first(row, "Name", "Security Name", "Description")
        action = first(row, "Action")
        quantity = self._amount(first(row, "Quantity", "Qty"))
        price = self._amount(first(row, "Price"))
        amount = self._amount(first(row, "Amount"))
        fees = self._amount(first(row, "Fees", "Fee", "Commission"))
        transaction_type = first(row, "Type", "Transaction Type")
        if not any(
            [
                transaction_id,
                trade_date_text,
                settlement_date_text,
                symbol,
                name,
                action,
                quantity is not None,
                price is not None,
                amount is not None,
                fees is not None,
                transaction_type,
            ]
        ):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "trade_date": timestamp.date().isoformat() if timestamp else trade_date_text,
                "timestamp": timestamp.isoformat() if timestamp else trade_date_text,
                "settlement_date": settlement_date.date().isoformat() if settlement_date else settlement_date_text,
                "symbol": symbol,
                "name": name,
                "action": action,
                "quantity": quantity,
                "price": price,
                "amount": amount,
                "fees": fees,
                "type": transaction_type,
                "transaction_id": transaction_id,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"webull_brokerage_transactions_csv:{transaction_id}" if transaction_id else digest_source_id(
            "webull_brokerage_transactions_csv",
            trade_date_text,
            settlement_date_text,
            symbol,
            name,
            action,
            amount,
            transaction_type,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="webull_brokerage_transactions_csv",
            source_id=source_id,
            source_entity_type="brokerage_transaction",
            title=self._title(action, symbol, name, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "webull", "brokerage_transaction", action, symbol, transaction_type] if tag)),
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

    def _title(self, action: str, symbol: str, name: str, amount: float | None) -> str:
        title = " ".join(part for part in [action, symbol] if part) or name or "Webull brokerage transaction"
        if amount is not None:
            return f"{title} ({amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Action: {metadata.get('action')}" if metadata.get("action") else "",
            f"Symbol: {metadata.get('symbol')}" if metadata.get("symbol") else "",
            f"Name: {metadata.get('name')}" if metadata.get("name") else "",
            f"Quantity: {metadata.get('quantity')}" if metadata.get("quantity") is not None else "",
            f"Price: {metadata.get('price')}" if metadata.get("price") is not None else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Fees: {metadata.get('fees')}" if metadata.get("fees") is not None else "",
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Trade date: {metadata.get('trade_date')}" if metadata.get("trade_date") else "",
            f"Settlement date: {metadata.get('settlement_date')}" if metadata.get("settlement_date") else "",
            f"Transaction ID: {metadata.get('transaction_id')}" if metadata.get("transaction_id") else "",
        ]
        return "\n".join(part for part in parts if part)
