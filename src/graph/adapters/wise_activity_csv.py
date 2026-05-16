"""Adapter for Wise activity CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class WiseActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "wise_activity_csv"

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
        transfer_id = first(row, "Transfer ID", "Transfer Id", "Transaction ID", "ID")
        date_text = first(row, "Date", "Created on", "Created", "Timestamp")
        updated_text = first(row, "Updated", "Updated on", "Completed on", "Date")
        timestamp = parse_datetime(date_text)
        updated_at = parse_datetime(updated_text) or timestamp
        activity_type = first(row, "Type", "Activity Type")
        status = first(row, "Status")
        reference = first(row, "Reference", "Description", "Payment reference")
        source_amount = parse_float(first(row, "Source Amount", "Source amount", "Amount"))
        source_currency = first(row, "Source Currency", "Source currency", "Currency")
        target_amount = parse_float(first(row, "Target Amount", "Target amount", "Received Amount"))
        target_currency = first(row, "Target Currency", "Target currency", "Received Currency")
        fee = parse_float(first(row, "Fee", "Fee Amount"))
        exchange_rate = parse_float(first(row, "Exchange Rate", "Rate"))
        recipient = first(row, "Recipient", "Recipient Name", "To")
        account = first(row, "Account", "Balance Account", "Profile")
        if not any([transfer_id, date_text, activity_type, reference, source_amount is not None, target_amount is not None, recipient]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transfer_id": transfer_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "updated_at": updated_at.isoformat() if updated_at else updated_text,
                "type": activity_type,
                "status": status,
                "reference": reference,
                "source_amount": source_amount,
                "source_currency": source_currency,
                "target_amount": target_amount,
                "target_currency": target_currency,
                "fee": fee,
                "exchange_rate": exchange_rate,
                "recipient": recipient,
                "account": account,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.WISE_ACTIVITY_CSV,
            source_id=f"wise_activity_csv:{transfer_id}" if transfer_id else digest_source_id("wise_activity_csv", date_text, activity_type, reference, source_amount, target_amount, index),
            source_entity_type="transaction",
            title=self._title(activity_type, recipient, source_amount, source_currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["wise", "transaction", activity_type, status, source_currency, target_currency] if tag)),
            created_at=timestamp or now,
            updated_at=updated_at or timestamp or now,
        )

    def _title(self, activity_type: str, recipient: str, amount: float | None, currency: str) -> str:
        parts = [part for part in [activity_type, recipient] if part]
        if amount is not None:
            parts.append(f"{amount:g} {currency}".strip())
        return " - ".join(parts) or "Wise activity"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Reference: {metadata.get('reference')}" if metadata.get("reference") else "",
            f"Recipient: {metadata.get('recipient')}" if metadata.get("recipient") else "",
            f"Source: {metadata.get('source_amount')} {metadata.get('source_currency', '')}".strip() if metadata.get("source_amount") is not None else "",
            f"Target: {metadata.get('target_amount')} {metadata.get('target_currency', '')}".strip() if metadata.get("target_amount") is not None else "",
            f"Fee: {metadata.get('fee')}" if metadata.get("fee") is not None else "",
        ]
        return "\n".join(part for part in parts if part)
