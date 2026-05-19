"""Adapter for Binance transaction history CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class BinanceTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "binance_transactions_csv"

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
        transaction_id = first(row, "Transaction ID", "Transaction Id", "Tx ID", "ID")
        order_id = first(row, "Order ID", "Order Id")
        time_text = first(row, "UTC_Time", "UTC Time", "Time", "Date", "Timestamp")
        timestamp = parse_datetime(time_text)
        account = first(row, "Account")
        operation = first(row, "Operation", "Type")
        coin = first(row, "Coin", "Asset", "Currency")
        change = parse_float(first(row, "Change", "Amount"))
        remark = first(row, "Remark", "Remarks", "Description")
        if not any([transaction_id, order_id, time_text, account, operation, coin, change is not None, remark]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "order_id": order_id,
                "utc_timestamp": timestamp.isoformat() if timestamp else time_text,
                "account": account,
                "operation": operation,
                "coin": coin,
                "change": change,
                "remark": remark,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        identifier = transaction_id or order_id
        source_id = f"binance_transactions_csv:{identifier}" if identifier else digest_source_id(
            "binance_transactions_csv",
            time_text,
            account,
            operation,
            coin,
            change,
            remark,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="binance_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(operation, coin, change),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "crypto", "binance", "transaction", account, operation, coin] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _title(self, operation: str, coin: str, change: float | None) -> str:
        title = " ".join(part for part in [operation, coin] if part) or "Binance transaction"
        if change is not None:
            return f"{title} ({change:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Operation: {metadata.get('operation')}" if metadata.get("operation") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Coin: {metadata.get('coin')}" if metadata.get("coin") else "",
            f"Change: {metadata.get('change')}" if metadata.get("change") is not None else "",
            f"Remark: {metadata.get('remark')}" if metadata.get("remark") else "",
            f"UTC timestamp: {metadata.get('utc_timestamp')}" if metadata.get("utc_timestamp") else "",
            f"Transaction ID: {metadata.get('transaction_id')}" if metadata.get("transaction_id") else "",
            f"Order ID: {metadata.get('order_id')}" if metadata.get("order_id") else "",
        ]
        return "\n".join(part for part in parts if part)
