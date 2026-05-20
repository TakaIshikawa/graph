"""Adapter for Quicken Simplifi transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SimplifiTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "simplifi_transactions_csv"

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
        date_text = first(row, "Date", "Transaction Date")
        timestamp = parse_datetime(date_text)
        payee = first(row, "Payee", "Description", "Name")
        category = first(row, "Category")
        tags = split_values(first(row, "Tags", "Tag"))
        account = first(row, "Account", "Account Name")
        amount = parse_money(first(row, "Amount"))
        notes = first(row, "Notes", "Memo")
        status = first(row, "Status", "State")
        transaction_type = first(row, "Type", "Transaction Type")
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID")
        if not any([transaction_id, date_text, payee, category, tags, account, amount is not None, notes, status, transaction_type]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "payee": payee,
                "category": category,
                "tags": tags,
                "account": account,
                "amount": amount,
                "notes": notes,
                "status": status,
                "type": transaction_type,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="simplifi_transactions_csv",
            source_id=f"simplifi_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("simplifi_transactions_csv", date_text, payee, amount, account, index),
            source_entity_type="transaction",
            title=self._title(payee, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "simplifi", category, account, status, transaction_type, *tags] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _title(self, payee: str, amount: float | None) -> str:
        if amount is not None:
            return f"{payee or 'Simplifi transaction'} ({amount:g})"
        return payee or "Simplifi transaction"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Payee: {metadata.get('payee')}" if metadata.get("payee") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Notes: {metadata.get('notes')}" if metadata.get("notes") else "",
        ]
        return "\n".join(part for part in parts if part)
