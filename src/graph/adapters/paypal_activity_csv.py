"""Adapter for PayPal activity CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class PaypalActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "paypal_activity_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["activity"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "activity" not in entity_types:
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
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID", "Txn ID")
        date_text = first(row, "Date", "Timestamp", "Time", "Datetime")
        timestamp = self._timestamp(row, date_text)
        name = first(row, "Name", "From Email Address", "To Email Address", "Counterparty")
        activity_type = first(row, "Type", "Transaction Type")
        status = first(row, "Status")
        note = first(row, "Subject", "Note", "Item Title", "Description")
        currency = first(row, "Currency", "Currency Code")
        gross = self._amount(first(row, "Gross", "Amount", "Total"))
        fee = self._amount(first(row, "Fee"))
        net = self._amount(first(row, "Net"))
        balance = self._amount(first(row, "Balance", "Running Balance"))
        reference = first(row, "Reference Txn ID", "Reference ID", "Invoice Number", "Receipt ID")
        if not any([transaction_id, date_text, name, activity_type, note, gross is not None, net is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "name": name,
                "type": activity_type,
                "status": status,
                "subject": first(row, "Subject"),
                "note": note,
                "gross": gross,
                "net": net,
                "fee": fee,
                "currency": currency,
                "balance": balance,
                "balance_impact": net if net is not None else gross,
                "reference": reference,
                "from_email": first(row, "From Email Address"),
                "to_email": first(row, "To Email Address"),
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.PAYPAL_ACTIVITY_CSV,
            source_id=f"paypal_activity_csv:{transaction_id}" if transaction_id else digest_source_id("paypal_activity_csv", date_text, name, activity_type, gross, net, index),
            source_entity_type="activity",
            title=self._title(name, activity_type, gross, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["paypal", "activity", activity_type, status, currency] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _timestamp(self, row: dict[str, Any], date_text: str) -> datetime | None:
        time_text = first(row, "Time")
        if date_text and time_text and time_text not in date_text:
            parsed = parse_datetime(f"{date_text} {time_text}")
            if parsed:
                return parsed
        return parse_datetime(date_text)

    def _amount(self, value: str) -> float | None:
        text = value.strip()
        if not text:
            return None
        negative = text.startswith("(") and text.endswith(")")
        cleaned = re.sub(r"[^0-9.\-]", "", text)
        if cleaned in {"", "-", "."}:
            return None
        try:
            amount = float(cleaned)
        except ValueError:
            return None
        return -abs(amount) if negative else amount

    def _title(self, name: str, activity_type: str, amount: float | None, currency: str) -> str:
        parts = [part for part in [activity_type, name] if part]
        if amount is not None:
            parts.append(f"{amount:g} {currency}".strip())
        return " - ".join(parts) or "PayPal activity"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Name: {metadata.get('name')}" if metadata.get("name") else "",
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Amount: {metadata.get('gross')} {metadata.get('currency', '')}".strip() if metadata.get("gross") is not None else "",
            f"Fee: {metadata.get('fee')}" if metadata.get("fee") is not None else "",
            f"Net: {metadata.get('net')}" if metadata.get("net") is not None else "",
            f"Note: {metadata.get('note')}" if metadata.get("note") else "",
        ]
        return "\n".join(part for part in parts if part)
