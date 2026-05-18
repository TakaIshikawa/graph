"""Adapter for Ally Bank transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AllyBankTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "ally_bank_transactions_csv"

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
        time_text = first(row, "Time", "Transaction Time")
        timestamp = parse_datetime(" ".join(part for part in [date_text, time_text] if part)) or parse_datetime(date_text)
        raw_amount = self._money(first(row, "Amount", "Transaction Amount"))
        transaction_type = first(row, "Type", "Transaction Type")
        amount = self._signed_amount(raw_amount, transaction_type)
        description = first(row, "Description", "Transaction Description", "Memo")
        check_number = first(row, "Check Number", "Check #", "Check No")
        balance = self._money(first(row, "Balance", "Running Balance"))
        account = first(row, "Account", "Account Name", "Account Number")
        if not any([date_text, time_text, raw_amount is not None, transaction_type, description, check_number, balance is not None, account]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "date": timestamp.date().isoformat() if timestamp else date_text,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "time": time_text,
                "amount": amount,
                "balance": balance,
                "type": transaction_type,
                "description": description,
                "check_number": check_number,
                "account": account,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="ally_bank_transactions_csv",
            source_id=digest_source_id("ally_bank_transactions_csv", date_text, time_text, description, transaction_type, amount, balance, account, check_number, index),
            source_entity_type="transaction",
            title=self._title(description, transaction_type, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["ally", "transaction", account, transaction_type] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _money(self, value: str) -> float | None:
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

    def _signed_amount(self, amount: float | None, transaction_type: str) -> float | None:
        if amount is None:
            return None
        normalized_type = transaction_type.casefold()
        if any(term in normalized_type for term in ["withdrawal", "debit", "payment", "check", "fee", "purchase"]):
            return -abs(amount)
        if any(term in normalized_type for term in ["deposit", "credit", "interest", "refund"]):
            return abs(amount)
        return amount

    def _title(self, description: str, transaction_type: str, amount: float | None) -> str:
        title = description or transaction_type or "Ally Bank transaction"
        if amount is not None:
            return f"{title} ({amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            metadata.get("description", ""),
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Balance: {metadata.get('balance')}" if metadata.get("balance") is not None else "",
            f"Check number: {metadata.get('check_number')}" if metadata.get("check_number") else "",
        ]
        return "\n".join(part for part in parts if part)
