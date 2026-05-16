"""Adapter for Coinbase transaction history CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class CoinbaseTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "coinbase_transactions_csv"

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
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID", "Order ID")
        timestamp_text = first(row, "Timestamp", "Date", "Time", "Created At")
        timestamp = parse_datetime(timestamp_text)
        transaction_type = first(row, "Transaction Type", "Type")
        asset = first(row, "Asset", "Asset Symbol", "Currency", "Currency/BTC")
        quantity = parse_float(first(row, "Quantity Transacted", "Quantity", "Amount", "Asset Amount"))
        spot_price = parse_float(first(row, "Spot Price at Transaction", "Spot Price", "Price"))
        subtotal = parse_float(first(row, "Subtotal", "Subtotal Amount"))
        total = parse_float(first(row, "Total (inclusive of fees and/or spread)", "Total", "Total Amount"))
        fees = parse_float(first(row, "Fees and/or Spread", "Fee", "Fees"))
        notes = first(row, "Notes", "Description", "Details")
        currency = first(row, "Subtotal Currency", "Total Currency", "Fiat Currency", "Currency Code")
        if not any([transaction_id, timestamp_text, transaction_type, asset, quantity is not None, total is not None, notes]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else timestamp_text,
                "transaction_type": transaction_type,
                "asset": asset,
                "quantity": quantity,
                "spot_price": spot_price,
                "subtotal": subtotal,
                "total": total,
                "fees": fees,
                "currency": currency,
                "notes": notes,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.COINBASE_TRANSACTIONS_CSV,
            source_id=f"coinbase_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("coinbase_transactions_csv", timestamp_text, transaction_type, asset, quantity, total, index),
            source_entity_type="transaction",
            title=self._title(transaction_type, asset, quantity),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["coinbase", "crypto", asset, transaction_type] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _title(self, transaction_type: str, asset: str, quantity: float | None) -> str:
        title = " ".join(part for part in [transaction_type, asset] if part) or "Coinbase transaction"
        if quantity is not None:
            return f"{title} ({quantity:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Type: {metadata.get('transaction_type')}" if metadata.get("transaction_type") else "",
            f"Asset: {metadata.get('asset')}" if metadata.get("asset") else "",
            f"Quantity: {metadata.get('quantity')}" if metadata.get("quantity") is not None else "",
            f"Spot price: {metadata.get('spot_price')}" if metadata.get("spot_price") is not None else "",
            f"Total: {metadata.get('total')} {metadata.get('currency', '')}".strip() if metadata.get("total") is not None else "",
            f"Fees: {metadata.get('fees')}" if metadata.get("fees") is not None else "",
            f"Notes: {metadata.get('notes')}" if metadata.get("notes") else "",
        ]
        return "\n".join(part for part in parts if part)
