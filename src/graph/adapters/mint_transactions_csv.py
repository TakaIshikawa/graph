"""Adapter for Mint transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class MintTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "mint_transactions_csv"

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
        description = first(row, "Description", "Name")
        original_description = first(row, "Original Description", "Original")
        raw_amount = parse_money(first(row, "Amount"))
        transaction_type = first(row, "Transaction Type", "Type")
        amount = self._signed_amount(raw_amount, transaction_type)
        category = first(row, "Category")
        account = first(row, "Account Name", "Account")
        labels = split_values(first(row, "Labels", "Tags"))
        notes = first(row, "Notes", "Memo")
        institution = first(row, "Institution", "Financial Institution")
        pending = self._bool(first(row, "Pending", "Is Pending"))
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID")
        if not any([transaction_id, date_text, description, original_description, raw_amount is not None, category, account, labels, notes, institution, pending is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "description": description,
                "original_description": original_description,
                "category": category,
                "account": account,
                "labels": labels,
                "notes": notes,
                "institution": institution,
                "pending": pending,
                "transaction_type": transaction_type,
                "amount": amount,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="mint_transactions_csv",
            source_id=f"mint_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("mint_transactions_csv", date_text, description, original_description, account, amount, index),
            source_entity_type="transaction",
            title=self._title(description, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["mint", "transaction", category, account, *labels] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _signed_amount(self, amount: float | None, transaction_type: str) -> float | None:
        if amount is None:
            return None
        if transaction_type.casefold() == "debit":
            return -abs(amount)
        if transaction_type.casefold() == "credit":
            return abs(amount)
        return amount

    def _bool(self, value: str) -> bool | None:
        text = value.strip().casefold()
        if not text:
            return None
        if text in {"true", "yes", "y", "1", "pending"}:
            return True
        if text in {"false", "no", "n", "0", "cleared"}:
            return False
        return None

    def _title(self, description: str, amount: float | None) -> str:
        if amount is not None:
            return f"{description or 'Mint transaction'} ({amount:g})"
        return description or "Mint transaction"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            metadata.get("description", ""),
            f"Original: {metadata.get('original_description')}" if metadata.get("original_description") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Notes: {metadata.get('notes')}" if metadata.get("notes") else "",
        ]
        return "\n".join(part for part in parts if part)
