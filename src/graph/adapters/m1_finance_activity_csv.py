"""Adapter for M1 Finance activity CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class M1FinanceActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "m1_finance_activity_csv"

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
        date_text = first(row, "Date", "Activity Date", "Trade Date", "Transaction Date")
        timestamp = parse_datetime(date_text)
        account = first(row, "Account", "Account Name", "Portfolio")
        symbol = first(row, "Symbol", "Ticker")
        security = first(row, "Security Name", "Security", "Name")
        action = first(row, "Action", "Type", "Activity Type", "Transaction Type")
        quantity = parse_money(first(row, "Quantity", "Qty", "Shares"))
        price = parse_money(first(row, "Price", "Share Price"))
        amount = parse_money(first(row, "Amount", "Net Amount", "Value"))
        cash_balance = parse_money(first(row, "Cash Balance", "Balance", "Running Cash Balance"))
        notes = first(row, "Notes", "Note", "Description", "Memo")
        activity_id = first(row, "Activity ID", "Transaction ID", "Transaction Id", "ID")
        if not any([activity_id, date_text, account, symbol, security, action, quantity is not None, price is not None, amount is not None, cash_balance is not None, notes]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "activity_id": activity_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "account": account,
                "symbol": symbol,
                "security": security,
                "action": action,
                "quantity": quantity,
                "price": price,
                "amount": amount,
                "cash_balance": cash_balance,
                "notes": notes,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"m1_finance_activity_csv:{activity_id}" if activity_id else digest_source_id(
            "m1_finance_activity_csv",
            date_text,
            account,
            symbol,
            security,
            action,
            amount,
            index,
        )
        return KnowledgeUnit(
            source_project="m1_finance_activity_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(action, symbol or security or notes, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "m1_finance", action, account, symbol] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _title(self, action: str, subject: str, amount: float | None) -> str:
        title = " ".join(part for part in [action, subject] if part) or "M1 Finance activity"
        if amount is not None:
            return f"{title} ({amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Action: {metadata.get('action')}" if metadata.get("action") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Symbol: {metadata.get('symbol')}" if metadata.get("symbol") else "",
            f"Security: {metadata.get('security')}" if metadata.get("security") else "",
            f"Quantity: {metadata.get('quantity')}" if metadata.get("quantity") is not None else "",
            f"Price: {metadata.get('price')}" if metadata.get("price") is not None else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Cash Balance: {metadata.get('cash_balance')}" if metadata.get("cash_balance") is not None else "",
            f"Notes: {metadata.get('notes')}" if metadata.get("notes") else "",
        ]
        return "\n".join(part for part in parts if part)
