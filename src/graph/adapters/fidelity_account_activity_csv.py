"""Adapter for Fidelity account activity CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class FidelityAccountActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "fidelity_account_activity_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["account_activity"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "account_activity" not in entity_types:
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
        run_date_text = first(row, "Run Date", "Date")
        settlement_date_text = first(row, "Settlement Date", "Settle Date")
        run_date = parse_datetime(run_date_text)
        settlement_date = parse_datetime(settlement_date_text)
        account = first(row, "Account", "Account Number")
        action = first(row, "Action", "Transaction Type", "Type")
        symbol = first(row, "Symbol")
        description = first(row, "Description")
        quantity = self._amount(first(row, "Quantity", "Qty"))
        price = self._amount(first(row, "Price", "Price ($)"))
        commission = self._amount(first(row, "Commission", "Commission ($)"))
        fees = self._amount(first(row, "Fees", "Fees ($)"))
        amount = self._amount(first(row, "Amount", "Amount ($)"))
        reference_number = first(row, "Reference Number", "Reference", "Activity ID")
        if not any([run_date_text, settlement_date_text, account, action, symbol, description, quantity is not None, price is not None, commission is not None, fees is not None, amount is not None, reference_number]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "run_date": self._date(run_date, run_date_text),
                "settlement_date": self._date(settlement_date, settlement_date_text),
                "account": account,
                "action": action,
                "symbol": symbol,
                "description": description,
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "fees": fees,
                "amount": amount,
                "reference_number": reference_number,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        timestamp = run_date or settlement_date or now
        return KnowledgeUnit(
            source_project="fidelity_account_activity_csv",
            source_id=f"fidelity_account_activity_csv:{reference_number}" if reference_number else digest_source_id("fidelity_account_activity_csv", run_date_text, settlement_date_text, account, action, symbol, description, amount, index),
            source_entity_type="account_activity",
            title=self._title(action, symbol, description, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "fidelity", "account_activity", action, symbol] if tag)),
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

    def _title(self, action: str, symbol: str, description: str, amount: float | None) -> str:
        title = " ".join(part for part in [action, symbol] if part) or description or "Fidelity account activity"
        if amount is not None:
            return f"{title} ({amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Action: {metadata.get('action')}" if metadata.get("action") else "",
            f"Symbol: {metadata.get('symbol')}" if metadata.get("symbol") else "",
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Quantity: {metadata.get('quantity')}" if metadata.get("quantity") is not None else "",
            f"Price: {metadata.get('price')}" if metadata.get("price") is not None else "",
            f"Commission: {metadata.get('commission')}" if metadata.get("commission") is not None else "",
            f"Fees: {metadata.get('fees')}" if metadata.get("fees") is not None else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Run date: {metadata.get('run_date')}" if metadata.get("run_date") else "",
            f"Settlement date: {metadata.get('settlement_date')}" if metadata.get("settlement_date") else "",
            f"Reference: {metadata.get('reference_number')}" if metadata.get("reference_number") else "",
        ]
        return "\n".join(part for part in parts if part)
