"""Adapter for Gemini transaction history CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GeminiTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "gemini_transactions_csv"

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
        date_text = first(row, "Date", "Timestamp", "Time")
        timestamp = parse_datetime(date_text)
        transaction_type = first(row, "Type", "Transaction Type")
        symbol = first(row, "Symbol", "Currency", "Asset")
        specification = first(row, "Specification")
        liquidity_indicator = first(row, "Liquidity Indicator", "Liquidity")
        trading_fee = parse_float(first(row, "Trading Fee", "TradingFee"))
        usd_amount = parse_float(first(row, "USD Amount", "Usd Amount", "USD"))
        amount = parse_float(first(row, "Amount"))
        fee = parse_float(first(row, "Fee", "Fees"))
        balance = parse_float(first(row, "Balance"))
        if not any(
            [
                transaction_id,
                date_text,
                transaction_type,
                symbol,
                specification,
                liquidity_indicator,
                trading_fee is not None,
                usd_amount is not None,
                amount is not None,
                fee is not None,
                balance is not None,
            ]
        ):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "transaction_type": transaction_type,
                "symbol": symbol,
                "specification": specification,
                "liquidity_indicator": liquidity_indicator,
                "trading_fee": trading_fee,
                "usd_amount": usd_amount,
                "amount": amount,
                "fee": fee,
                "balance": balance,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"gemini_transactions_csv:{transaction_id}" if transaction_id else digest_source_id(
            "gemini_transactions_csv",
            date_text,
            transaction_type,
            symbol,
            specification,
            liquidity_indicator,
            usd_amount,
            amount,
            fee,
            balance,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="gemini_transactions_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(transaction_type, symbol, amount, usd_amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "crypto", "gemini", "transaction", symbol, transaction_type] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _title(self, transaction_type: str, symbol: str, amount: float | None, usd_amount: float | None) -> str:
        title = " ".join(part for part in [transaction_type, symbol] if part) or "Gemini transaction"
        if amount is not None:
            return f"{title} ({amount:g})"
        if usd_amount is not None:
            return f"{title} ({usd_amount:g} USD)"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Type: {metadata.get('transaction_type')}" if metadata.get("transaction_type") else "",
            f"Symbol: {metadata.get('symbol')}" if metadata.get("symbol") else "",
            f"Specification: {metadata.get('specification')}" if metadata.get("specification") else "",
            f"Liquidity: {metadata.get('liquidity_indicator')}" if metadata.get("liquidity_indicator") else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"USD amount: {metadata.get('usd_amount')}" if metadata.get("usd_amount") is not None else "",
            f"Trading fee: {metadata.get('trading_fee')}" if metadata.get("trading_fee") is not None else "",
            f"Fee: {metadata.get('fee')}" if metadata.get("fee") is not None else "",
            f"Balance: {metadata.get('balance')}" if metadata.get("balance") is not None else "",
            f"Timestamp: {metadata.get('timestamp')}" if metadata.get("timestamp") else "",
            f"Transaction ID: {metadata.get('transaction_id')}" if metadata.get("transaction_id") else "",
        ]
        return "\n".join(part for part in parts if part)
