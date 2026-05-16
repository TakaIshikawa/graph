"""Adapter for Revolut transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class RevolutTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "revolut_transactions_csv"

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
        completed_text = first(row, "Completed Date", "Completed", "Date completed")
        started_text = first(row, "Started Date", "Started", "Date started", "Date")
        timestamp = parse_datetime(completed_text) or parse_datetime(started_text)
        description = first(row, "Description", "Reference", "Merchant")
        category = first(row, "Category", "Type")
        currency = first(row, "Currency", "Currency Code")
        paid_out = parse_money(first(row, "Paid Out", "Paid out", "Out"))
        paid_in = parse_money(first(row, "Paid In", "Paid in", "In"))
        exchange_out = parse_money(first(row, "Exchange Out", "Exchange out"))
        exchange_in = parse_money(first(row, "Exchange In", "Exchange in"))
        balance = parse_money(first(row, "Balance", "Running Balance"))
        amount = self._amount(paid_out, paid_in, first(row, "Amount"))
        account = {
            "account_id": first(row, "Account ID", "Account Id"),
            "account_name": first(row, "Account", "Account Name"),
            "account_type": first(row, "Account Type"),
        }
        if not any([transaction_id, completed_text, started_text, description, category, amount is not None, balance is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "completed_at": timestamp.isoformat() if timestamp and completed_text else completed_text,
                "started_at": parse_datetime(started_text).isoformat() if parse_datetime(started_text) else started_text,
                "description": description,
                "category": category,
                "paid_out": paid_out,
                "paid_in": paid_in,
                "exchange_out": exchange_out,
                "exchange_in": exchange_in,
                "amount": amount,
                "balance": balance,
                "currency": currency,
                **clean_metadata(account),
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.REVOLUT_TRANSACTIONS_CSV,
            source_id=f"revolut_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("revolut_transactions_csv", completed_text, started_text, description, amount, index),
            source_entity_type="transaction",
            title=self._title(description, category, amount, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["revolut", "transaction", category, currency] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _amount(self, paid_out: float | None, paid_in: float | None, amount_text: str) -> float | None:
        amount = parse_money(amount_text)
        if amount is not None:
            return amount
        if paid_in is not None:
            return paid_in
        if paid_out is not None:
            return -abs(paid_out)
        return None

    def _title(self, description: str, category: str, amount: float | None, currency: str) -> str:
        title = description or category or "Revolut transaction"
        if amount is not None:
            return f"{title} ({amount:g} {currency})".strip()
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            metadata.get("description", ""),
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Balance: {metadata.get('balance')} {metadata.get('currency', '')}".strip() if metadata.get("balance") is not None else "",
            f"Account: {metadata.get('account_name')}" if metadata.get("account_name") else "",
        ]
        return "\n".join(part for part in parts if part)
