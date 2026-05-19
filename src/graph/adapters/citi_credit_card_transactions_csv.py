"""Adapter for Citi credit card transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class CitiCreditCardTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "citi_credit_card_transactions_csv"

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
        status = first(row, "Status")
        date_text = first(row, "Date", "Transaction Date")
        timestamp = parse_datetime(date_text)
        description = first(row, "Description", "Merchant")
        debit = self._amount(first(row, "Debit"))
        credit = self._amount(first(row, "Credit"))
        explicit_amount = self._amount(first(row, "Amount"))
        amount = self._signed_amount(debit, credit, explicit_amount)
        category = first(row, "Category")
        member_name = first(row, "Member Name", "Card Member", "Cardmember")
        account = first(row, "Account", "Account Name")
        reference = first(row, "Reference Number", "Reference", "Transaction ID", "Transaction Id", "ID")
        if not date_text or not description:
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "date": self._date(timestamp, date_text),
                "description": description,
                "category": category,
                "status": status,
                "member_name": member_name,
                "account": account,
                "debit": debit,
                "credit": credit,
                "amount": amount,
                "currency": "USD" if amount is not None else "",
                "reference": reference,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"citi_credit_card_transactions_csv:{reference}" if reference else digest_source_id(
            "citi_credit_card_transactions_csv",
            date_text,
            description,
            category,
            status,
            member_name,
            account,
            debit,
            credit,
            explicit_amount,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="citi_credit_card_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(description, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "citi", "credit-card", category, status] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _signed_amount(self, debit: float | None, credit: float | None, explicit_amount: float | None) -> float | None:
        if debit is not None:
            return -abs(debit)
        if credit is not None:
            return abs(credit)
        return explicit_amount

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

    def _title(self, description: str, amount: float | None) -> str:
        title = description or "Citi credit card transaction"
        if amount is not None:
            return f"{title} ({amount:g} USD)"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Debit: {metadata.get('debit')}" if metadata.get("debit") is not None else "",
            f"Credit: {metadata.get('credit')}" if metadata.get("credit") is not None else "",
            f"Date: {metadata.get('date')}" if metadata.get("date") else "",
            f"Member name: {metadata.get('member_name')}" if metadata.get("member_name") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Reference: {metadata.get('reference')}" if metadata.get("reference") else "",
        ]
        return "\n".join(part for part in parts if part)
