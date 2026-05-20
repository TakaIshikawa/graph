"""Adapter for Rocket Money transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class RocketMoneyTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "rocket_money_transactions_csv"

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
        account = first(row, "Account", "Account Name")
        category = first(row, "Category")
        amount = parse_money(first(row, "Amount"))
        currency = first(row, "Currency", "Currency Code") or "USD"
        status = first(row, "Status", "State")
        notes = first(row, "Notes", "Memo")
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID")
        pending = self._pending(first(row, "Pending", "Is Pending", "Status"))
        if not any([transaction_id, date_text, description, account, category, amount is not None, status, notes, pending is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "description": description,
                "merchant": description,
                "account": account,
                "category": category,
                "amount": amount,
                "currency": currency,
                "status": status,
                "pending": pending,
                "notes": notes,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="rocket_money_transactions_csv",
            source_id=f"rocket_money_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("rocket_money_transactions_csv", date_text, description, account, category, amount, index),
            source_entity_type="transaction",
            title=self._title(description, amount, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "rocket_money", category, account, status] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _pending(self, value: str) -> bool | None:
        text = value.strip().casefold()
        if text in {"pending", "true", "yes", "1"}:
            return True
        if text in {"cleared", "posted", "false", "no", "0"}:
            return False
        return None

    def _title(self, description: str, amount: float | None, currency: str) -> str:
        title = description or "Rocket Money transaction"
        if amount is not None:
            return f"{title} ({amount:g}{(' ' + currency) if currency else ''})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Notes: {metadata.get('notes')}" if metadata.get("notes") else "",
        ]
        return "\n".join(part for part in parts if part)
