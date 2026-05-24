"""Adapter for QuickBooks transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class QuickBooksTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "quickbooks_transactions_csv"

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
                if unit is None or (sync_at and ("timestamp" not in unit.metadata or unit.updated_at <= sync_at)):
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        date_text = first(row, "Date", "Transaction Date")
        timestamp = parse_datetime(date_text)
        transaction_type = first(row, "Transaction Type", "Type")
        number = first(row, "Num", "Number", "No.")
        name = first(row, "Name", "Customer", "Vendor", "Payee")
        memo = first(row, "Memo", "Description")
        account = first(row, "Account")
        split = first(row, "Split")
        debit = parse_money(first(row, "Debit"))
        credit = parse_money(first(row, "Credit"))
        amount = self._amount(debit, credit, first(row, "Amount"))
        balance = parse_money(first(row, "Balance"))
        klass = first(row, "Class")
        location = first(row, "Location")
        if not any([date_text, transaction_type, number, name, memo, account, split, debit is not None, credit is not None, amount is not None, balance is not None, klass, location]):
            return None
        now = datetime.now(timezone.utc)
        occurred_at = timestamp or now
        metadata = clean_metadata(
            {
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "transaction_type": transaction_type,
                "number": number,
                "name": name,
                "memo": memo,
                "account": account,
                "split": split,
                "debit": debit,
                "credit": credit,
                "amount": amount,
                "balance": balance,
                "class": klass,
                "location": location,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="quickbooks_transactions_csv",
            source_id=digest_source_id("quickbooks_transactions_csv", date_text, transaction_type, number, name, account, amount, index),
            source_entity_type="transaction",
            title=self._title(transaction_type, name, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["quickbooks", "transaction", transaction_type, account, klass, location] if tag)),
            created_at=occurred_at,
            updated_at=occurred_at,
        )

    def _amount(self, debit: float | None, credit: float | None, amount_text: str) -> float | None:
        amount = parse_money(amount_text)
        if amount is not None:
            return amount
        if credit is not None:
            return credit
        if debit is not None:
            return -abs(debit)
        return None

    def _title(self, transaction_type: str, name: str, amount: float | None) -> str:
        title = " - ".join(part for part in [transaction_type, name] if part) or "QuickBooks transaction"
        return f"{title} ({amount:g})" if amount is not None else title

    def _content(self, metadata: dict[str, Any]) -> str:
        labels = [("Type", "transaction_type"), ("Name", "name"), ("Memo", "memo"), ("Account", "account"), ("Split", "split"), ("Amount", "amount")]
        return "\n".join(f"{label}: {metadata[key]}" for label, key in labels if key in metadata)
