"""Adapter for US Bank transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class UsBankTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "us_bank_transactions_csv"

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
        timestamp = parse_datetime(date_text)
        transaction = first(row, "Transaction", "Transaction Type", "Type")
        name = first(row, "Name", "Payee", "Description")
        memo = first(row, "Memo", "Notes")
        category = first(row, "Category")
        account = first(row, "Account", "Account Name")
        amount_text = first(row, "Amount")
        amount = self._signed_amount(self._amount(amount_text), transaction)
        debit = self._amount(first(row, "Debit"))
        credit = self._amount(first(row, "Credit"))
        if debit is not None:
            amount = -abs(debit)
        if credit is not None:
            amount = abs(credit)
        balance = self._amount(first(row, "Balance", "Running Balance", "Running Bal."))
        reference = first(row, "Reference Number", "Reference", "Transaction ID", "Transaction Id", "ID")
        if not any([date_text, transaction, name, memo, amount is not None, debit is not None, credit is not None, balance is not None, category, account, reference]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "date": self._date(timestamp, date_text),
                "transaction": transaction,
                "name": name,
                "memo": memo,
                "amount": amount,
                "debit": debit,
                "credit": credit,
                "balance": balance,
                "currency": "USD" if amount is not None else "",
                "category": category,
                "account": account,
                "reference": reference,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"us_bank_transactions_csv:{reference}" if reference else digest_source_id(
            "us_bank_transactions_csv",
            date_text,
            transaction,
            name,
            memo,
            amount_text,
            debit,
            credit,
            balance,
            category,
            account,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="us_bank_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(name, transaction, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "us-bank", category, transaction] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _signed_amount(self, amount: float | None, transaction: str) -> float | None:
        if amount is None:
            return None
        if amount < 0:
            return amount
        normalized_transaction = transaction.casefold()
        if any(word in normalized_transaction for word in ("deposit", "payment", "credit", "refund", "return")):
            return abs(amount)
        if any(word in normalized_transaction for word in ("withdrawal", "debit", "purchase", "charge", "sale")):
            return -abs(amount)
        return amount

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

    def _title(self, name: str, transaction: str, amount: float | None) -> str:
        title = name or transaction or "US Bank transaction"
        if amount is not None:
            return f"{title} ({amount:g} USD)"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Name: {metadata.get('name')}" if metadata.get("name") else "",
            f"Transaction: {metadata.get('transaction')}" if metadata.get("transaction") else "",
            f"Memo: {metadata.get('memo')}" if metadata.get("memo") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Debit: {metadata.get('debit')}" if metadata.get("debit") is not None else "",
            f"Credit: {metadata.get('credit')}" if metadata.get("credit") is not None else "",
            f"Balance: {metadata.get('balance')}" if metadata.get("balance") is not None else "",
            f"Date: {metadata.get('date')}" if metadata.get("date") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Reference: {metadata.get('reference')}" if metadata.get("reference") else "",
        ]
        return "\n".join(part for part in parts if part)
