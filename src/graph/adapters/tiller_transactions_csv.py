"""Adapter for Tiller Money transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class TillerTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "tiller_transactions_csv"

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
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID")
        date_text = first(row, "Date", "Transaction Date", "Posted Date")
        timestamp = parse_datetime(date_text)
        description = first(row, "Description", "Name")
        full_description = first(row, "Full Description", "Original Description", "Details")
        category = first(row, "Category")
        amount = self._amount(first(row, "Amount"))
        account = first(row, "Account", "Account Name")
        account_number = first(row, "Account #", "Account Number", "Account No")
        institution = first(row, "Institution", "Bank", "Financial Institution")
        month = first(row, "Month")
        week = first(row, "Week")
        check_number = first(row, "Check Number", "Check #", "Check No")
        if not any([transaction_id, date_text, description, full_description, category, amount is not None, account, account_number, institution, month, week, check_number]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "date": timestamp.date().isoformat() if timestamp else date_text,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "description": description,
                "full_description": full_description,
                "category": category,
                "amount": amount,
                "account": account,
                "account_number": account_number,
                "institution": institution,
                "month": month,
                "week": week,
                "check_number": check_number,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="tiller_transactions_csv",
            source_id=f"tiller_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("tiller_transactions_csv", date_text, description, full_description, category, amount, account, account_number, institution, check_number, index),
            source_entity_type="transaction",
            title=self._title(description, full_description, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["tiller", "transaction", account, institution, category] if tag)),
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

    def _title(self, description: str, full_description: str, amount: float | None) -> str:
        title = description or full_description or "Tiller transaction"
        if amount is not None:
            return f"{title} ({amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            metadata.get("description", ""),
            f"Full description: {metadata.get('full_description')}" if metadata.get("full_description") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Institution: {metadata.get('institution')}" if metadata.get("institution") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Check number: {metadata.get('check_number')}" if metadata.get("check_number") else "",
        ]
        return "\n".join(part for part in parts if part)
