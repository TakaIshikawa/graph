"""Adapter for YNAB budget transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class YnabTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "ynab_transactions_csv"

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
        account = first(row, "Account", "Account Name")
        date_text = first(row, "Date", "Transaction Date")
        timestamp = parse_datetime(date_text)
        payee = first(row, "Payee")
        category = first(row, "Category Group/Category", "Category")
        memo = first(row, "Memo", "Notes")
        outflow = parse_money(first(row, "Outflow"))
        inflow = parse_money(first(row, "Inflow"))
        amount = self._amount(outflow, inflow, first(row, "Amount"))
        cleared = first(row, "Cleared", "Status")
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID")
        if not any([transaction_id, date_text, account, payee, category, memo, outflow is not None, inflow is not None, amount is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "account": account,
                "payee": payee,
                "category": category,
                "memo": memo,
                "outflow": outflow,
                "inflow": inflow,
                "amount": amount,
                "cleared": cleared,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.YNAB_TRANSACTIONS_CSV,
            source_id=f"ynab_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("ynab_transactions_csv", date_text, account, payee, category, amount, index),
            source_entity_type="transaction",
            title=self._title(payee, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["ynab", "transaction", account, category, cleared] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _amount(self, outflow: float | None, inflow: float | None, amount_text: str) -> float | None:
        amount = parse_money(amount_text)
        if amount is not None:
            return amount
        if inflow is not None:
            return inflow
        if outflow is not None:
            return -abs(outflow)
        return None

    def _title(self, payee: str, amount: float | None) -> str:
        if amount is not None:
            return f"{payee or 'YNAB transaction'} ({amount:g})"
        return payee or "YNAB transaction"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Payee: {metadata.get('payee')}" if metadata.get("payee") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Memo: {metadata.get('memo')}" if metadata.get("memo") else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Cleared: {metadata.get('cleared')}" if metadata.get("cleared") else "",
        ]
        return "\n".join(part for part in parts if part)
