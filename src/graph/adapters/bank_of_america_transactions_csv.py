"""Adapter for Bank of America transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class BankOfAmericaTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "bank_of_america_transactions_csv"

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
                rows = self._read_rows(path)
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

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            lines = handle.readlines()
        header_index = 0
        for index, line in enumerate(lines):
            normalized = {cell.strip().casefold() for cell in next(csv.reader([line]))}
            has_date = bool({"date", "posted date", "transaction date"} & normalized)
            has_description = bool({"description", "payee"} & normalized)
            if has_date and has_description and "amount" in normalized:
                header_index = index
                break
        reader = csv.DictReader(lines[header_index:])
        if not reader.fieldnames:
            return []
        return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        posted_date_text = first(row, "Posted Date", "Date", "Transaction Date")
        timestamp = parse_datetime(posted_date_text)
        payee = first(row, "Payee")
        address = first(row, "Address")
        description = first(row, "Description")
        memo = first(row, "Memo", "Notes")
        amount = self._amount(first(row, "Amount"))
        running_balance = self._amount(first(row, "Running Bal.", "Running Balance", "Balance"))
        account = first(row, "Account", "Account Name")
        category = first(row, "Category")
        status = first(row, "Status")
        transaction_type = first(row, "Transaction Type", "Type")
        reference_number = first(row, "Reference Number", "Reference")
        if not any([posted_date_text, payee, address, description, memo, amount is not None, running_balance is not None, account, category, status, transaction_type, reference_number]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "posted_date": self._date(timestamp, posted_date_text),
                "date": self._date(timestamp, posted_date_text),
                "reference_number": reference_number,
                "payee": payee,
                "address": address,
                "description": description,
                "memo": memo,
                "amount": amount,
                "currency": "USD" if amount is not None else "",
                "running_balance": running_balance,
                "account": account,
                "category": category,
                "status": status,
                "transaction_type": transaction_type,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"bank_of_america_transactions_csv:{reference_number}" if reference_number else digest_source_id(
            "bank_of_america_transactions_csv",
            posted_date_text,
            payee,
            address,
            description,
            memo,
            amount,
            running_balance,
            account,
            category,
            status,
            transaction_type,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="bank_of_america_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(description, transaction_type, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "bank-of-america", transaction_type, status] if tag)),
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

    def _title(self, description: str, transaction_type: str, amount: float | None) -> str:
        title = description or transaction_type or "Bank of America transaction"
        if amount is not None:
            return f"{title} ({amount:g} USD)"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Payee: {metadata.get('payee')}" if metadata.get("payee") else "",
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Memo: {metadata.get('memo')}" if metadata.get("memo") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Type: {metadata.get('transaction_type')}" if metadata.get("transaction_type") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Running balance: {metadata.get('running_balance')}" if metadata.get("running_balance") is not None else "",
            f"Posted date: {metadata.get('posted_date')}" if metadata.get("posted_date") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Address: {metadata.get('address')}" if metadata.get("address") else "",
            f"Reference: {metadata.get('reference_number')}" if metadata.get("reference_number") else "",
        ]
        return "\n".join(part for part in parts if part)
