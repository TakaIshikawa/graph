"""Adapter for Xero bank transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class XeroBankTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "xero_bank_transactions_csv"

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
        payee = first(row, "Payee", "Contact", "Name")
        description = first(row, "Description", "Details")
        reference = first(row, "Reference", "Ref")
        account = first(row, "Account", "Bank Account")
        spent = parse_money(first(row, "Spent", "Withdrawal", "Debit"))
        received = parse_money(first(row, "Received", "Deposit", "Credit"))
        amount = self._amount(spent, received, first(row, "Amount"))
        tax = parse_money(first(row, "Tax", "Tax Amount", "GST"))
        currency = first(row, "Currency")
        status = first(row, "Status", "Reconciliation Status")
        reconciled = first(row, "Reconciled", "Is Reconciled")
        if not any([date_text, payee, description, reference, account, spent is not None, received is not None, amount is not None, tax is not None, currency, status, reconciled]):
            return None
        now = datetime.now(timezone.utc)
        occurred_at = timestamp or now
        metadata = clean_metadata(
            {
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "payee": payee,
                "description": description,
                "reference": reference,
                "account": account,
                "spent": spent,
                "received": received,
                "amount": amount,
                "tax": tax,
                "currency": currency,
                "status": status,
                "reconciled": reconciled,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="xero_bank_transactions_csv",
            source_id=digest_source_id("xero_bank_transactions_csv", date_text, payee, description, reference, account, amount, index),
            source_entity_type="transaction",
            title=self._title(payee, description, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["xero", "transaction", account, status, reconciled] if tag)),
            created_at=occurred_at,
            updated_at=occurred_at,
        )

    def _amount(self, spent: float | None, received: float | None, amount_text: str) -> float | None:
        amount = parse_money(amount_text)
        if amount is not None:
            return amount
        if received is not None:
            return received
        if spent is not None:
            return -abs(spent)
        return None

    def _title(self, payee: str, description: str, amount: float | None) -> str:
        title = payee or description or "Xero bank transaction"
        return f"{title} ({amount:g})" if amount is not None else title

    def _content(self, metadata: dict[str, Any]) -> str:
        labels = [("Payee", "payee"), ("Description", "description"), ("Reference", "reference"), ("Account", "account"), ("Amount", "amount"), ("Status", "status")]
        return "\n".join(f"{label}: {metadata[key]}" for label, key in labels if key in metadata)
