"""Adapter for American Express transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AmexTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "amex_transactions_csv"

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
        card_member = first(row, "Card Member", "Cardmember")
        account_number = first(row, "Account #", "Account Number")
        amount = self._amount(first(row, "Amount"))
        extended_details = first(row, "Extended Details")
        statement_descriptor = first(row, "Appears On Your Statement As", "Appears on Your Statement As")
        address = first(row, "Address")
        city_state = first(row, "City/State", "City State")
        zip_code = first(row, "Zip Code", "ZIP Code")
        country = first(row, "Country")
        reference = first(row, "Reference", "Reference Number")
        category = first(row, "Category")
        if not any([date_text, description, card_member, account_number, amount is not None, statement_descriptor, reference, category]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "date": self._date(timestamp, date_text),
                "description": description,
                "card_member": card_member,
                "account_number": account_number,
                "amount": amount,
                "currency": "USD" if amount is not None else "",
                "extended_details": extended_details,
                "statement_descriptor": statement_descriptor,
                "address": address,
                "city_state": city_state,
                "zip_code": zip_code,
                "country": country,
                "reference": reference,
                "category": category,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = digest_source_id(
            "amex_transactions_csv",
            date_text,
            description,
            card_member,
            account_number,
            amount,
            statement_descriptor,
            reference,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="amex_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(description, statement_descriptor, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "amex", category] if tag)),
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

    def _title(self, description: str, statement_descriptor: str, amount: float | None) -> str:
        title = description or statement_descriptor or "Amex transaction"
        if amount is not None:
            return f"{title} ({amount:g} USD)"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Statement descriptor: {metadata.get('statement_descriptor')}" if metadata.get("statement_descriptor") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Date: {metadata.get('date')}" if metadata.get("date") else "",
            f"Card member: {metadata.get('card_member')}" if metadata.get("card_member") else "",
            f"Reference: {metadata.get('reference')}" if metadata.get("reference") else "",
        ]
        return "\n".join(part for part in parts if part)
