"""Adapter for Acorns activity CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AcornsActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "acorns_activity_csv"

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
        date_text = first(row, "Date", "Activity Date", "Transaction Date", "Created At")
        timestamp = parse_datetime(date_text)
        account = first(row, "Account", "Account Name")
        activity_type = first(row, "Activity Type", "Type", "Transaction Type")
        description = first(row, "Description", "Details")
        merchant = first(row, "Merchant", "Merchant Name")
        category = first(row, "Category")
        round_up_amount = self._amount(first(row, "Round-Up Amount", "Round Up Amount", "Roundup Amount"))
        investment_amount = self._amount(first(row, "Investment Amount", "Invested Amount"))
        total_amount = self._amount(first(row, "Total Amount", "Amount", "Transaction Amount"))
        status = first(row, "Status", "State")
        transaction_id = first(row, "Transaction ID", "Transaction Id", "Activity ID", "ID")
        if not any(
            [
                date_text,
                account,
                activity_type,
                description,
                merchant,
                category,
                round_up_amount is not None,
                investment_amount is not None,
                total_amount is not None,
                status,
                transaction_id,
            ]
        ):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "date": timestamp.date().isoformat() if timestamp else date_text,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "account": account,
                "activity_type": activity_type,
                "description": description,
                "merchant": merchant,
                "category": category,
                "round_up_amount": round_up_amount,
                "investment_amount": investment_amount,
                "total_amount": total_amount,
                "currency": "USD" if any(amount is not None for amount in [round_up_amount, investment_amount, total_amount]) else "",
                "status": status,
                "transaction_id": transaction_id,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        source_id = f"acorns_activity_csv:{transaction_id}" if transaction_id else digest_source_id(
            "acorns_activity_csv",
            date_text,
            account,
            activity_type,
            description,
            merchant,
            category,
            round_up_amount,
            investment_amount,
            total_amount,
            index,
        )
        timestamp = timestamp or now
        return KnowledgeUnit(
            source_project="acorns_activity_csv",
            source_id=source_id,
            source_entity_type="transaction",
            title=self._title(activity_type, merchant, description, total_amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["finance", "transaction", "acorns", activity_type, category, status] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _amount(self, value: str) -> float | None:
        text = value.strip()
        if not text:
            return None
        negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
        cleaned = re.sub(r"[^0-9.]", "", text)
        if cleaned in {"", "."}:
            return None
        try:
            amount = float(cleaned)
        except ValueError:
            return None
        return -abs(amount) if negative else amount

    def _title(self, activity_type: str, merchant: str, description: str, total_amount: float | None) -> str:
        title = " - ".join(part for part in [activity_type, merchant or description] if part) or "Acorns activity"
        if total_amount is not None:
            return f"{title} ({total_amount:g} USD)"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Activity Type: {metadata.get('activity_type')}" if metadata.get("activity_type") else "",
            f"Merchant: {metadata.get('merchant')}" if metadata.get("merchant") else "",
            f"Description: {metadata.get('description')}" if metadata.get("description") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Round-Up Amount: {metadata.get('round_up_amount')}" if metadata.get("round_up_amount") is not None else "",
            f"Investment Amount: {metadata.get('investment_amount')}" if metadata.get("investment_amount") is not None else "",
            f"Total Amount: {metadata.get('total_amount')}" if metadata.get("total_amount") is not None else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Date: {metadata.get('date')}" if metadata.get("date") else "",
            f"Account: {metadata.get('account')}" if metadata.get("account") else "",
            f"Transaction ID: {metadata.get('transaction_id')}" if metadata.get("transaction_id") else "",
        ]
        return "\n".join(part for part in parts if part)
