"""Adapter for Robinhood account activity CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class RobinhoodAccountActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "robinhood_account_activity_csv"

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
        activity_date_text = first(row, "Activity Date", "Date", "Trade Date")
        process_date_text = first(row, "Process Date", "Settlement Date", "Settle Date")
        activity_date = parse_datetime(activity_date_text)
        process_date = parse_datetime(process_date_text)
        symbol = first(row, "Symbol", "Instrument", "Ticker")
        description = first(row, "Description", "Name")
        activity_type = first(row, "Activity Type", "Type", "Trans Code", "Transaction Type")
        quantity = parse_float(first(row, "Quantity", "Qty", "Shares"))
        price = parse_float(first(row, "Price", "Share Price"))
        amount = parse_float(first(row, "Amount", "Net Amount", "Total"))
        fees = parse_float(first(row, "Fees", "Fee", "Regulatory Fees"))
        currency = first(row, "Currency", "ISO Currency Code")
        account = first(row, "Account", "Account Name", "Account Number")
        transaction_id = first(row, "Transaction ID", "Transaction Id", "Activity ID", "Activity Id", "ID")

        if not any([activity_date_text, process_date_text, symbol, description, activity_type, quantity is not None, price is not None, amount is not None, fees is not None, currency, account, transaction_id]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "activity_date": activity_date.isoformat() if activity_date else activity_date_text,
                "process_date": process_date.isoformat() if process_date else process_date_text,
                "symbol": symbol,
                "description": description,
                "activity_type": activity_type,
                "quantity": quantity,
                "price": price,
                "amount": amount,
                "fees": fees,
                "currency": currency,
                "account": account,
                "source_file": source_file,
            }
        )

        return KnowledgeUnit(
            source_project="robinhood_account_activity_csv",
            source_id=f"robinhood_account_activity_csv:{transaction_id}" if transaction_id else digest_source_id("robinhood_account_activity_csv", activity_date_text, process_date_text, symbol, description, activity_type, amount, index),
            source_entity_type="account_activity",
            title=self._title(activity_type, symbol, amount, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["robinhood", "account_activity", activity_type, symbol] if tag)),
            created_at=activity_date or process_date or now,
            updated_at=process_date or activity_date or now,
        )

    def _title(self, activity_type: str, symbol: str, amount: float | None, currency: str) -> str:
        title = " ".join(part for part in [activity_type, symbol] if part) or "Robinhood account activity"
        if amount is not None:
            suffix = f"{amount:g} {currency}".strip()
            return f"{title} ({suffix})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = []
        for key, label in (
            ("activity_date", "Activity date"),
            ("process_date", "Process date"),
            ("activity_type", "Type"),
            ("symbol", "Symbol"),
            ("description", "Description"),
            ("quantity", "Quantity"),
            ("price", "Price"),
            ("amount", "Amount"),
            ("fees", "Fees"),
            ("account", "Account"),
        ):
            if metadata.get(key) is not None:
                value = metadata[key]
                if key in {"price", "amount", "fees"} and metadata.get("currency"):
                    value = f"{value} {metadata['currency']}"
                parts.append(f"{label}: {value}")
        return "\n".join(parts)
