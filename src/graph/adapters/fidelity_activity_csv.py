"""Adapter for Fidelity brokerage activity CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class FidelityActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "fidelity_activity_csv"

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
        run_date_text = first(row, "Run Date", "Date")
        run_date = parse_datetime(run_date_text)
        settlement_date_text = first(row, "Settlement Date", "Settle Date")
        settlement_date = parse_datetime(settlement_date_text)
        action = first(row, "Action", "Transaction Type", "Type")
        symbol = first(row, "Symbol")
        description = first(row, "Description")
        quantity = parse_money(first(row, "Quantity", "Qty"))
        price = parse_money(first(row, "Price ($)", "Price"))
        amount = parse_money(first(row, "Amount ($)", "Amount"))
        activity_id = first(row, "Activity ID", "ID")
        if not any([activity_id, run_date_text, settlement_date_text, action, symbol, description, quantity is not None, price is not None, amount is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "activity_id": activity_id,
                "run_date": run_date.isoformat() if run_date else run_date_text,
                "settlement_date": settlement_date.isoformat() if settlement_date else settlement_date_text,
                "action": action,
                "symbol": symbol,
                "description": description,
                "quantity": quantity,
                "price": price,
                "amount": amount,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.FIDELITY_ACTIVITY_CSV,
            source_id=f"fidelity_activity_csv:{activity_id}" if activity_id else digest_source_id("fidelity_activity_csv", run_date_text, settlement_date_text, action, symbol, description, amount, index),
            source_entity_type="brokerage_activity",
            title=self._title(action, symbol, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["fidelity", "brokerage_activity", action, symbol] if tag)),
            created_at=run_date or settlement_date or now,
            updated_at=run_date or settlement_date or now,
        )

    def _title(self, action: str, symbol: str, amount: float | None) -> str:
        title = " ".join(part for part in [action, symbol] if part) or "Fidelity activity"
        if amount is not None:
            return f"{title} ({amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Action: {metadata.get('action')}" if metadata.get("action") else "",
            f"Symbol: {metadata.get('symbol')}" if metadata.get("symbol") else "",
            metadata.get("description", ""),
            f"Quantity: {metadata.get('quantity')}" if metadata.get("quantity") is not None else "",
            f"Price: {metadata.get('price')}" if metadata.get("price") is not None else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
        ]
        return "\n".join(part for part in parts if part)
