"""Adapter for Personal Capital transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PersonalCapitalTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "personal_capital_transactions_csv"

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
        account = first(row, "Account", "Account Name")
        description = first(row, "Description", "Merchant", "Payee")
        original_description = first(row, "Original Description", "Original Merchant", "Original Name")
        category = first(row, "Category")
        amount = parse_money(first(row, "Amount", "Transaction Amount"))
        tags = split_values(first(row, "Tags", "Labels"))
        notes = first(row, "Notes", "Note", "Memo")
        transaction_id = first(row, "Transaction ID", "Transaction Id", "Activity ID", "ID")
        if not any([transaction_id, date_text, account, description, original_description, category, amount is not None, tags, notes]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "account": account,
                "merchant": description,
                "description": description,
                "original_description": original_description,
                "category": category,
                "amount": amount,
                "tags": tags,
                "notes": notes,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"personal_capital_transactions_csv:{transaction_id}" if transaction_id else digest_source_id(
            "personal_capital_transactions_csv",
            date_text,
            description,
            amount,
            account,
            index,
        )
        return KnowledgeUnit(
            source_project="personal_capital_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(description, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "personal_capital", category, account, *tags] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _title(self, description: str, amount: float | None) -> str:
        title = description or "Personal Capital transaction"
        if amount is not None:
            return f"{title} ({amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Original Description: {metadata.get('original_description')}" if metadata.get("original_description") else "",
            f"Notes: {metadata.get('notes')}" if metadata.get("notes") else "",
        ]
        return "\n".join(part for part in parts if part)
