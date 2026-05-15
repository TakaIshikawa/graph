"""Adapter for Monzo transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class MonzoTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "monzo_transactions_csv"

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
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID", "id")
        date_text = first(row, "Date", "Created", "Created at", "Datetime", "Timestamp")
        timestamp = parse_datetime(date_text)
        description = first(row, "Description", "Name")
        merchant = first(row, "Merchant", "Merchant Name")
        category = first(row, "Category")
        amount = parse_money(first(row, "Amount", "Amount (GBP)", "Total"))
        currency = first(row, "Currency", "Currency Code")
        local_amount = parse_money(first(row, "Local Amount", "Local amount"))
        local_currency = first(row, "Local Currency", "Local currency")
        notes = first(row, "Notes", "Note")
        tags = [tag.removeprefix("#") for tag in split_values(first(row, "Tags", "Labels"))]
        address = {
            "address": first(row, "Address"),
            "city": first(row, "City"),
            "postcode": first(row, "Postcode", "Postal Code"),
            "country": first(row, "Country"),
            "latitude": parse_money(first(row, "Latitude", "Lat")),
            "longitude": parse_money(first(row, "Longitude", "Lon", "Lng")),
        }
        account = {
            "account_id": first(row, "Account ID", "Account Id"),
            "account_name": first(row, "Account", "Account Name"),
            "account_type": first(row, "Account Type"),
        }
        if not any([transaction_id, date_text, description, merchant, amount is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "description": description,
                "merchant": merchant,
                "category": category,
                "amount": amount,
                "currency": currency,
                "local_amount": local_amount,
                "local_currency": local_currency,
                "notes": notes,
                "tags": tags,
                "address": clean_metadata(address) or None,
                **clean_metadata(account),
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.MONZO_TRANSACTIONS_CSV,
            source_id=f"monzo_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("monzo_transactions_csv", date_text, description, merchant, amount, index),
            source_entity_type="transaction",
            title=self._title(description, merchant, amount, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["monzo", "transaction", category, *tags] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _title(self, description: str, merchant: str, amount: float | None, currency: str) -> str:
        title = merchant or description or "Monzo transaction"
        if amount is not None:
            return f"{title} ({amount:g} {currency})".strip()
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            metadata.get("description", ""),
            f"Merchant: {metadata.get('merchant')}" if metadata.get("merchant") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Notes: {metadata.get('notes')}" if metadata.get("notes") else "",
        ]
        return "\n".join(part for part in parts if part)
