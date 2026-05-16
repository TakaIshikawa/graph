"""Adapter for Stripe balance transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class StripeBalanceTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "stripe_balance_transactions_csv"

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
        transaction_id = first(row, "id", "ID", "Balance Transaction ID", "Balance Transaction")
        created_text = first(row, "Created", "created", "Created (UTC)", "Date")
        available_text = first(row, "Available On", "Available on", "Available Date")
        created_at = parse_datetime(created_text)
        available_at = parse_datetime(available_text)
        transaction_type = first(row, "Type", "type")
        description = first(row, "Description", "description")
        amount = parse_float(first(row, "Amount", "amount"))
        fee = parse_float(first(row, "Fee", "fee"))
        net = parse_float(first(row, "Net", "net"))
        currency = first(row, "Currency", "currency")
        source = first(row, "Source", "source")
        payout = first(row, "Payout", "payout")
        reporting_category = first(row, "Reporting Category", "reporting_category")
        status = first(row, "Status", "status")
        if not any([transaction_id, created_text, transaction_type, description, amount is not None, fee is not None, net is not None, source]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "created_at": created_at.isoformat() if created_at else created_text,
                "available_at": available_at.isoformat() if available_at else available_text,
                "type": transaction_type,
                "description": description,
                "amount": amount,
                "fee": fee,
                "net": net,
                "currency": currency,
                "source": source,
                "payout": payout,
                "reporting_category": reporting_category,
                "status": status,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.STRIPE_BALANCE_TRANSACTIONS_CSV,
            source_id=f"stripe_balance_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("stripe_balance_transactions_csv", created_text, transaction_type, amount, net, source, index),
            source_entity_type="transaction",
            title=self._title(transaction_type, description, amount, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["stripe", "transaction", transaction_type, reporting_category, status, currency] if tag)),
            created_at=created_at or now,
            updated_at=created_at or now,
        )

    def _title(self, transaction_type: str, description: str, amount: float | None, currency: str) -> str:
        title = " - ".join(part for part in [transaction_type, description] if part) or "Stripe balance transaction"
        if amount is not None:
            return f"{title} ({amount:g} {currency})".strip()
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            metadata.get("description", ""),
            f"Reporting category: {metadata.get('reporting_category')}" if metadata.get("reporting_category") else "",
            f"Amount: {metadata.get('amount')} {metadata.get('currency', '')}".strip() if metadata.get("amount") is not None else "",
            f"Fee: {metadata.get('fee')}" if metadata.get("fee") is not None else "",
            f"Net: {metadata.get('net')}" if metadata.get("net") is not None else "",
        ]
        return "\n".join(part for part in parts if part)
