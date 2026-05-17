"""Adapter for Cash App activity CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class CashAppActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "cash_app_activity_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["activity", "transaction"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and not {"activity", "transaction"}.intersection(entity_types):
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
        date_text = first(row, "Date", "Transaction Date", "Created At")
        timestamp = parse_datetime(date_text)
        transaction_type = first(row, "Transaction Type", "Type", "Activity Type")
        name = first(row, "Name", "Display Name", "Cashtag")
        amount = self._amount(first(row, "Amount", "Net Amount"))
        currency = first(row, "Currency", "Currency Code")
        status = first(row, "Status")
        notes = first(row, "Notes", "Note", "Description")
        fee = self._amount(first(row, "Fee", "Fees"))
        asset_type = first(row, "Asset Type", "Asset")
        asset_price = self._amount(first(row, "Asset Price", "Asset Price USD", "Price"))
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID", "Identifier")
        if not any([transaction_id, date_text, transaction_type, name, amount is not None, status, notes, asset_type]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "transaction_type": transaction_type,
                "name": name,
                "amount": amount,
                "currency": currency,
                "status": status,
                "notes": notes,
                "fee": fee,
                "asset_type": asset_type,
                "asset_price": asset_price,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = (
            f"cash_app_activity_csv:{transaction_id}"
            if transaction_id
            else digest_source_id("cash_app_activity_csv", date_text, transaction_type, name, amount, currency, status, notes, index)
        )
        return KnowledgeUnit(
            source_project="cash_app_activity_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(transaction_type, name, amount, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "cash-app", transaction_type, status, asset_type] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
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

    def _title(self, transaction_type: str, name: str, amount: float | None, currency: str) -> str:
        parts = [part for part in [transaction_type, name] if part]
        if amount is not None:
            parts.append(f"{amount:g} {currency}".strip())
        return " - ".join(parts) or "Cash App activity"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Type: {metadata.get('transaction_type')}" if metadata.get("transaction_type") else "",
            f"Name: {metadata.get('name')}" if metadata.get("name") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Fee: {metadata.get('fee')}" if metadata.get("fee") is not None else "",
            f"Asset: {metadata.get('asset_type')}" if metadata.get("asset_type") else "",
            f"Asset price: {metadata.get('asset_price')}" if metadata.get("asset_price") is not None else "",
            f"Notes: {metadata.get('notes')}" if metadata.get("notes") else "",
        ]
        return "\n".join(part for part in parts if part)
