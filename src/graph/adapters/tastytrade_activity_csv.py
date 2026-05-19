"""Adapter for tastytrade activity CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class TastytradeActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "tastytrade_activity_csv"

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
        date_text = self._date_text(row)
        timestamp = parse_datetime(date_text)
        account = first(row, "Account", "Account Number", "Account Name")
        action = first(row, "Action", "Type", "Activity Type", "Transaction Type")
        symbol = first(row, "Symbol")
        underlying = first(row, "Underlying", "Underlying Symbol")
        description = first(row, "Description", "Details")
        quantity = self._amount(first(row, "Quantity", "Qty"))
        price = self._amount(first(row, "Price"))
        amount = self._amount(first(row, "Amount", "Net Amount"))
        commission = self._amount(first(row, "Commission", "Commissions"))
        fees = self._amount(first(row, "Fees", "Fee", "Regulatory Fees"))
        currency = first(row, "Currency")
        transaction_id = first(row, "Transaction ID", "Transaction Id", "Transaction Number", "ID")
        order_id = first(row, "Order ID", "Order Id", "Order Number")
        if not any(
            [
                date_text,
                account,
                action,
                symbol,
                underlying,
                description,
                quantity is not None,
                price is not None,
                amount is not None,
                commission is not None,
                fees is not None,
                currency,
                transaction_id,
                order_id,
            ]
        ):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "date": timestamp.date().isoformat() if timestamp else date_text,
                "account": account,
                "action": action,
                "symbol": symbol,
                "underlying": underlying,
                "description": description,
                "quantity": quantity,
                "price": price,
                "amount": amount,
                "commission": commission,
                "fees": fees,
                "currency": currency,
                "transaction_id": transaction_id,
                "order_id": order_id,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = self._source_id(transaction_id, order_id, date_text, account, action, symbol, description, amount, index)
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="tastytrade_activity_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(action, symbol, underlying, description, amount, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "tastytrade", action, symbol, underlying] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _date_text(self, row: dict[str, Any]) -> str:
        date_time = first(row, "Date/Time", "Date Time", "Datetime", "Timestamp")
        if date_time:
            return date_time
        date = first(row, "Date", "Activity Date", "Transaction Date")
        time = first(row, "Time")
        return f"{date} {time}".strip() if date and time else date

    def _source_id(self, transaction_id: str, order_id: str, date_text: str, account: str, action: str, symbol: str, description: str, amount: float | None, index: int) -> str:
        if transaction_id:
            return f"tastytrade_activity_csv:{transaction_id}"
        if order_id:
            return f"tastytrade_activity_csv:order:{order_id}"
        return digest_source_id("tastytrade_activity_csv", date_text, account, action, symbol, description, amount, index)

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

    def _title(self, action: str, symbol: str, underlying: str, description: str, amount: float | None, currency: str) -> str:
        instrument = symbol or underlying
        title = " ".join(part for part in [action, instrument] if part) or description or "tastytrade activity"
        if amount is not None:
            suffix = f" {currency}" if currency else ""
            return f"{title} ({amount:g}{suffix})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Action: {metadata.get('action')}" if metadata.get("action") else "",
            f"Symbol: {metadata.get('symbol')}" if metadata.get("symbol") else "",
            f"Underlying: {metadata.get('underlying')}" if metadata.get("underlying") else "",
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Quantity: {metadata.get('quantity')}" if metadata.get("quantity") is not None else "",
            f"Price: {metadata.get('price')}" if metadata.get("price") is not None else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Commission: {metadata.get('commission')}" if metadata.get("commission") is not None else "",
            f"Fees: {metadata.get('fees')}" if metadata.get("fees") is not None else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Date: {metadata.get('date')}" if metadata.get("date") else "",
            f"Transaction ID: {metadata.get('transaction_id')}" if metadata.get("transaction_id") else "",
            f"Order ID: {metadata.get('order_id')}" if metadata.get("order_id") else "",
        ]
        return "\n".join(part for part in parts if part)
