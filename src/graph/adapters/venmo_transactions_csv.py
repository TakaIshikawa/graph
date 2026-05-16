"""Adapter for Venmo transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class VenmoTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "venmo_transactions_csv"

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
        date_text = first(row, "Datetime", "Date", "Completed Date", "Timestamp")
        timestamp = parse_datetime(date_text)
        transaction_type = first(row, "Type", "Transaction Type")
        status = first(row, "Status")
        note = first(row, "Note", "Description", "Memo")
        from_person = first(row, "From", "From User")
        to_person = first(row, "To", "To User")
        amount = self._amount(first(row, "Amount", "Total"))
        fee = self._amount(first(row, "Fee"))
        funding_source = first(row, "Funding Source", "Funding")
        destination = first(row, "Destination", "Bank/Card", "Bank")
        privacy = first(row, "Privacy", "Audience")
        if not any([transaction_id, date_text, transaction_type, note, from_person, to_person, amount is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "type": transaction_type,
                "status": status,
                "note": note,
                "from": from_person,
                "to": to_person,
                "amount": amount,
                "fee": fee,
                "funding_source": funding_source,
                "destination": destination,
                "privacy": privacy,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.VENMO_TRANSACTIONS_CSV,
            source_id=f"venmo_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("venmo_transactions_csv", date_text, transaction_type, note, from_person, to_person, amount, index),
            source_entity_type="transaction",
            title=self._title(transaction_type, note, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["venmo", "transaction", transaction_type, status, privacy] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _amount(self, value: str) -> float | None:
        text = value.strip()
        if not text:
            return None
        negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
        cleaned = re.sub(r"[^0-9.]", "", text)
        if not cleaned:
            return None
        try:
            amount = float(cleaned)
        except ValueError:
            return None
        return -abs(amount) if negative else amount

    def _title(self, transaction_type: str, note: str, amount: float | None) -> str:
        parts = [part for part in [transaction_type, note] if part]
        if amount is not None:
            parts.append(f"{amount:g}")
        return " - ".join(parts) or "Venmo transaction"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Note: {metadata.get('note')}" if metadata.get("note") else "",
            f"From: {metadata.get('from')}" if metadata.get("from") else "",
            f"To: {metadata.get('to')}" if metadata.get("to") else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Fee: {metadata.get('fee')}" if metadata.get("fee") is not None else "",
        ]
        return "\n".join(part for part in parts if part)
