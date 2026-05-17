"""Adapter for Vanguard transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class VanguardTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "vanguard_transactions_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["investment_transaction"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "investment_transaction" not in entity_types:
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
        transaction_id = first(row, "Transaction ID", "Transaction Id", "Activity ID", "ID")
        trade_date_text = first(row, "Trade Date", "Transaction Date", "Date")
        trade_date = parse_datetime(trade_date_text)
        settlement_date_text = first(row, "Settlement Date", "Settle Date")
        settlement_date = parse_datetime(settlement_date_text)
        account = first(row, "Account", "Account Number", "Account Name")
        investment_name = first(row, "Investment Name", "Investment", "Security Name", "Fund Name")
        symbol = first(row, "Symbol", "Ticker")
        transaction_type = first(row, "Transaction Type", "Type", "Action")
        shares = parse_money(first(row, "Shares", "Quantity"))
        share_price = parse_money(first(row, "Share Price", "Price"))
        principal_amount = parse_money(first(row, "Principal Amount", "Principal"))
        commission_fees = parse_money(first(row, "Commission Fees", "Commission", "Fees", "Fee"))
        net_amount = parse_money(first(row, "Net Amount", "Amount", "Total"))
        if not any(
            [
                transaction_id,
                trade_date_text,
                settlement_date_text,
                account,
                investment_name,
                symbol,
                transaction_type,
                shares is not None,
                share_price is not None,
                principal_amount is not None,
                commission_fees is not None,
                net_amount is not None,
            ]
        ):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "trade_date": trade_date.isoformat() if trade_date else trade_date_text,
                "settlement_date": settlement_date.isoformat() if settlement_date else settlement_date_text,
                "account": account,
                "investment_name": investment_name,
                "symbol": symbol,
                "transaction_type": transaction_type,
                "shares": shares,
                "share_price": share_price,
                "principal_amount": principal_amount,
                "commission_fees": commission_fees,
                "net_amount": net_amount,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="vanguard_transactions_csv",
            source_id=f"vanguard_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("vanguard_transactions_csv", trade_date_text, settlement_date_text, account, investment_name, symbol, transaction_type, net_amount, index),
            source_entity_type="investment_transaction",
            title=self._title(transaction_type, symbol, investment_name, net_amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["vanguard", "investment_transaction", transaction_type, symbol, account] if tag)),
            created_at=trade_date or settlement_date or now,
            updated_at=trade_date or settlement_date or now,
        )

    def _title(self, transaction_type: str, symbol: str, investment_name: str, net_amount: float | None) -> str:
        title = " ".join(part for part in [transaction_type, symbol or investment_name] if part) or "Vanguard transaction"
        if net_amount is not None:
            return f"{title} ({net_amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Type: {metadata.get('transaction_type')}" if metadata.get("transaction_type") else "",
            f"Investment: {metadata.get('investment_name')}" if metadata.get("investment_name") else "",
            f"Symbol: {metadata.get('symbol')}" if metadata.get("symbol") else "",
            f"Shares: {metadata.get('shares')}" if metadata.get("shares") is not None else "",
            f"Share price: {metadata.get('share_price')}" if metadata.get("share_price") is not None else "",
            f"Principal amount: {metadata.get('principal_amount')}" if metadata.get("principal_amount") is not None else "",
            f"Commission fees: {metadata.get('commission_fees')}" if metadata.get("commission_fees") is not None else "",
            f"Net amount: {metadata.get('net_amount')}" if metadata.get("net_amount") is not None else "",
        ]
        return "\n".join(part for part in parts if part)
