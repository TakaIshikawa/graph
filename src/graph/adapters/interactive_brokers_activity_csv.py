"""Adapter for focused Interactive Brokers activity statement CSV rows."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class InteractiveBrokersActivityCsvAdapter(SourceAdapter):
    _SUPPORTED_MARKERS = (
        "trade",
        "dividend",
        "interest",
        "fee",
        "deposit",
        "withdrawal",
        "withholding",
        "tax",
        "cash transaction",
    )

    @property
    def name(self) -> str:
        return "interactive_brokers_activity_csv"

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
        section = first(row, "Section", "Statement Section", "DataDiscriminator")
        row_type = first(row, "Type", "Transaction Type", "Code", "Activity Type")
        description = first(row, "Description", "Transaction Description", "Details")
        if not self._is_supported(section, row_type, description):
            return None

        date_text = first(row, "Date", "Trade Date", "Transaction Date", "Settle Date", "Report Date")
        timestamp = parse_datetime(date_text)
        account = first(row, "Account", "Account ID", "Account Number")
        asset_class = first(row, "Asset Class", "AssetClass")
        symbol = first(row, "Symbol", "Ticker")
        quantity = parse_money(first(row, "Quantity", "Qty", "Shares"))
        price = parse_money(first(row, "Price", "Trade Price"))
        amount = self._amount(row)
        currency = first(row, "Currency", "Currency Code", "Proceeds Currency")
        commission = parse_money(first(row, "Commission", "Commissions", "Fee", "Fees"))
        activity_id = first(row, "Transaction ID", "Activity ID", "Trade ID", "ID")
        if not any([activity_id, date_text, account, asset_class, symbol, description, quantity is not None, price is not None, amount is not None, currency, commission is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "activity_id": activity_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "section": section,
                "type": row_type,
                "account": account,
                "asset_class": asset_class,
                "symbol": symbol,
                "description": description,
                "quantity": quantity,
                "price": price,
                "amount": amount,
                "currency": currency,
                "commission": commission,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"interactive_brokers_activity_csv:{activity_id}" if activity_id else digest_source_id(
            "interactive_brokers_activity_csv",
            section,
            date_text,
            account,
            symbol,
            description,
            amount,
            index,
        )
        return KnowledgeUnit(
            source_project="interactive_brokers_activity_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(row_type or section, symbol or description, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "interactive_brokers", section, row_type, asset_class, symbol] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _is_supported(self, section: str, row_type: str, description: str) -> bool:
        text = " ".join([section, row_type, description]).casefold()
        if any(marker in text for marker in ("summary", "header", "statement")):
            return False
        return any(marker in text for marker in self._SUPPORTED_MARKERS)

    def _amount(self, row: dict[str, Any]) -> float | None:
        for key in ("Amount", "Net Amount", "Proceeds", "Cash Amount", "Debit", "Credit"):
            amount = parse_money(first(row, key))
            if amount is not None:
                if key == "Debit":
                    return -abs(amount)
                return amount
        return None

    def _title(self, row_type: str, subject: str, amount: float | None) -> str:
        title = " ".join(part for part in [row_type, subject] if part) or "Interactive Brokers activity"
        if amount is not None:
            return f"{title} ({amount:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Section: {metadata.get('section')}" if metadata.get("section") else "",
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Asset Class: {metadata.get('asset_class')}" if metadata.get("asset_class") else "",
            f"Symbol: {metadata.get('symbol')}" if metadata.get("symbol") else "",
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Quantity: {metadata.get('quantity')}" if metadata.get("quantity") is not None else "",
            f"Price: {metadata.get('price')}" if metadata.get("price") is not None else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Commission: {metadata.get('commission')}" if metadata.get("commission") is not None else "",
        ]
        return "\n".join(part for part in parts if part)
