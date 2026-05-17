"""Adapter for generic Plaid transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PlaidTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "plaid_transactions_csv"

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
        transaction_id = first(row, "transaction_id", "Transaction ID", "Transaction Id", "ID")
        date_text = first(row, "date", "Date", "Transaction Date", "Posted Date")
        timestamp = parse_datetime(date_text)
        authorized_date_text = first(row, "authorized_date", "Authorized Date", "Auth Date")
        authorized_date = parse_datetime(authorized_date_text)
        name = first(row, "name", "Name", "Description")
        merchant_name = first(row, "merchant_name", "Merchant Name", "Merchant")
        amount = parse_float(first(row, "amount", "Amount"))
        currency = first(row, "iso_currency_code", "ISO Currency Code", "currency", "Currency")
        account_id = first(row, "account_id", "Account ID")
        account_name = first(row, "account_name", "Account Name", "Account")
        category = split_values(first(row, "category", "Category"))
        pending = self._parse_bool(first(row, "pending", "Pending"))
        payment_channel = first(row, "payment_channel", "Payment Channel", "Channel")
        location_city = first(row, "location_city", "Location City", "City")
        location_region = first(row, "location_region", "Location Region", "Region", "State")
        location_country = first(row, "location_country", "Location Country", "Country")
        website = first(row, "website", "Website", "URL")
        if not any([transaction_id, date_text, authorized_date_text, name, merchant_name, amount is not None, account_id, account_name, category, pending is not None, payment_channel, location_city, location_region, location_country, website]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "date": timestamp.isoformat() if timestamp else date_text,
                "authorized_date": authorized_date.isoformat() if authorized_date else authorized_date_text,
                "name": name,
                "merchant_name": merchant_name,
                "amount": amount,
                "currency": currency,
                "account_id": account_id,
                "account_name": account_name,
                "category": category,
                "pending": pending,
                "payment_channel": payment_channel,
                "location_city": location_city,
                "location_region": location_region,
                "location_country": location_country,
                "website": website,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="plaid_transactions_csv",
            source_id=f"plaid_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("plaid_transactions_csv", date_text, authorized_date_text, name, merchant_name, amount, account_id, index),
            source_entity_type="transaction",
            title=self._title(name, merchant_name, amount, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["plaid", "transaction", account_name, merchant_name, payment_channel, *category] if tag)),
            created_at=timestamp or authorized_date or now,
            updated_at=timestamp or authorized_date or now,
        )

    def _title(self, name: str, merchant_name: str, amount: float | None, currency: str) -> str:
        title = merchant_name or name or "Plaid transaction"
        if amount is not None:
            return f"{title} ({amount:g} {currency})".strip()
        return title

    def _parse_bool(self, value: str) -> bool | None:
        text = value.strip().casefold()
        if text in {"true", "t", "yes", "y", "1"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
        return None

    def _content(self, metadata: dict[str, Any]) -> str:
        category = metadata.get("category") or []
        parts = [
            metadata.get("name", ""),
            f"Merchant: {metadata.get('merchant_name')}" if metadata.get("merchant_name") else "",
            f"Account: {metadata.get('account_name')}" if metadata.get("account_name") else "",
            f"Category: {', '.join(category)}" if category else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Pending: {metadata.get('pending')}" if metadata.get("pending") is not None else "",
            f"Channel: {metadata.get('payment_channel')}" if metadata.get("payment_channel") else "",
            f"Location: {', '.join(part for part in [metadata.get('location_city'), metadata.get('location_region'), metadata.get('location_country')] if part)}" if any(metadata.get(key) for key in ["location_city", "location_region", "location_country"]) else "",
            f"Website: {metadata.get('website')}" if metadata.get("website") else "",
        ]
        return "\n".join(part for part in parts if part)
