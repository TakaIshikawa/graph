"""Adapter for Mercury banking transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class MercuryTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "mercury_transactions_csv"

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
        description = first(row, "Description")
        amount = self._amount(first(row, "Amount"))
        balance = self._amount(first(row, "Balance"))
        bank_description = first(row, "Bank Description")
        category = first(row, "Category")
        note = first(row, "Note", "Memo")
        status = first(row, "Status")
        counterparty_name = first(row, "Counterparty Name", "Counterparty")
        counterparty_account = first(row, "Counterparty Account")
        reference_id = first(row, "Reference ID", "Reference")
        if not any([date_text, description, amount is not None, balance is not None, bank_description, category, note, status, counterparty_name, counterparty_account, reference_id]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "date": self._date(timestamp, date_text),
                "description": description,
                "amount": amount,
                "currency": "USD" if amount is not None else "",
                "balance": balance,
                "bank_description": bank_description,
                "category": category,
                "note": note,
                "status": status,
                "counterparty_name": counterparty_name,
                "counterparty_account": counterparty_account,
                "reference_id": reference_id,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"mercury_transactions_csv:{reference_id}" if reference_id else digest_source_id(
            "mercury_transactions_csv",
            date_text,
            description,
            amount,
            balance,
            counterparty_name,
            counterparty_account,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="mercury_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(description, counterparty_name, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "mercury", category, status] if tag)),
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

    def _date(self, timestamp: datetime | None, fallback: str) -> str:
        return timestamp.date().isoformat() if timestamp else fallback

    def _title(self, description: str, counterparty_name: str, amount: float | None) -> str:
        title = description or counterparty_name or "Mercury transaction"
        if amount is not None:
            return f"{title} ({amount:g} USD)"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Counterparty: {metadata.get('counterparty_name')}" if metadata.get("counterparty_name") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Balance: {metadata.get('balance')}" if metadata.get("balance") is not None else "",
            f"Date: {metadata.get('date')}" if metadata.get("date") else "",
            f"Reference: {metadata.get('reference_id')}" if metadata.get("reference_id") else "",
            f"Note: {metadata.get('note')}" if metadata.get("note") else "",
        ]
        return "\n".join(part for part in parts if part)
