"""Adapter for Robinhood account activity CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class RobinhoodActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "robinhood_activity_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["brokerage_activity"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "brokerage_activity" not in entity_types:
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
        activity_date_text = first(row, "Activity Date", "Date")
        activity_date = parse_datetime(activity_date_text)
        process_date_text = first(row, "Process Date", "Settlement Date")
        process_date = parse_datetime(process_date_text)
        instrument = first(row, "Instrument", "Symbol")
        description = first(row, "Description")
        transaction_code = first(row, "Trans Code", "Transaction Code", "Type")
        quantity = parse_money(first(row, "Quantity", "Qty"))
        price = parse_money(first(row, "Price"))
        amount = parse_money(first(row, "Amount"))
        activity_id = first(row, "Activity ID", "ID")
        if not any([activity_id, activity_date_text, process_date_text, instrument, description, transaction_code, quantity is not None, price is not None, amount is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "activity_id": activity_id,
                "activity_date": activity_date.isoformat() if activity_date else activity_date_text,
                "process_date": process_date.isoformat() if process_date else process_date_text,
                "instrument": instrument,
                "description": description,
                "transaction_code": transaction_code,
                "quantity": quantity,
                "price": price,
                "amount": amount,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.ROBINHOOD_ACTIVITY_CSV,
            source_id=f"robinhood_activity_csv:{activity_id}" if activity_id else digest_source_id("robinhood_activity_csv", activity_date_text, process_date_text, instrument, description, transaction_code, amount, index),
            source_entity_type="brokerage_activity",
            title=self._title(transaction_code, instrument, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["robinhood", "brokerage_activity", instrument, transaction_code] if tag)),
            created_at=activity_date or process_date or now,
            updated_at=activity_date or process_date or now,
        )

    def _title(self, transaction_code: str, instrument: str, amount: float | None) -> str:
        title = " ".join(part for part in [transaction_code, instrument] if part) or "Robinhood activity"
        if amount is not None:
            return f"{title} ({amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Code: {metadata.get('transaction_code')}" if metadata.get("transaction_code") else "",
            f"Instrument: {metadata.get('instrument')}" if metadata.get("instrument") else "",
            metadata.get("description", ""),
            f"Quantity: {metadata.get('quantity')}" if metadata.get("quantity") is not None else "",
            f"Price: {metadata.get('price')}" if metadata.get("price") is not None else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
        ]
        return "\n".join(part for part in parts if part)
