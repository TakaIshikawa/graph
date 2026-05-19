"""Adapter for Wise activity CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class WiseActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "wise_activity_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["transaction", "currency"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or ["transaction"])
        if not allowed_types.intersection(self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        transactions: list[KnowledgeUnit] = []

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
                transactions.append(unit)
                if "transaction" in allowed_types:
                    result.units.append(unit)

        currencies = self._currency_units(transactions) if "currency" in allowed_types else []
        if "currency" in allowed_types:
            result.units.extend(currencies)
        if {"transaction", "currency"}.issubset(allowed_types):
            result.edges.extend(self._currency_edges(transactions, currencies))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
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

    def _currency_units(self, transactions: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for transaction in transactions:
            for currency in {self._normalize_currency(transaction.metadata.get("source_currency")), self._normalize_currency(transaction.metadata.get("target_currency"))}:
                if currency:
                    grouped.setdefault(currency, []).append(transaction)

        units: list[KnowledgeUnit] = []
        for currency, items in sorted(grouped.items()):
            sent_total = sum(
                float(item.metadata["source_amount"])
                for item in items
                if self._normalize_currency(item.metadata.get("source_currency")) == currency and isinstance(item.metadata.get("source_amount"), int | float)
            )
            received_total = sum(
                float(item.metadata["target_amount"])
                for item in items
                if self._normalize_currency(item.metadata.get("target_currency")) == currency and isinstance(item.metadata.get("target_amount"), int | float)
            )
            metadata = clean_metadata(
                {
                    "currency": currency,
                    "activity_count": len(items),
                    "first_seen": min(item.created_at for item in items).isoformat(),
                    "last_seen": max(item.created_at for item in items).isoformat(),
                    "sent_total": sent_total if sent_total else None,
                    "received_total": received_total if received_total else None,
                    "activity_source_ids": sorted(item.source_id for item in items),
                }
            )
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.WISE_ACTIVITY_CSV,
                    source_id=self._currency_source_id(currency),
                    source_entity_type="currency",
                    title=currency,
                    content=f"Wise currency: {currency}\nActivities: {len(items)}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=["wise", "currency", currency],
                    created_at=min(item.created_at for item in items),
                    updated_at=max(item.updated_at for item in items),
                )
            )
        return units

    def _currency_edges(self, transactions: list[KnowledgeUnit], currencies: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        currency_ids = {unit.metadata["currency"]: unit.source_id for unit in currencies}
        edges = []
        for transaction in transactions:
            for currency in {self._normalize_currency(transaction.metadata.get("source_currency")), self._normalize_currency(transaction.metadata.get("target_currency"))}:
                target = currency_ids.get(currency)
                if target:
                    edges.append(self._edge(transaction.source_id, target, "transaction_currency"))
        return list({edge.id: edge for edge in edges}.values())

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        digest = digest_source_id("wise_activity_csv:edge", from_id, relation_type, to_id).rsplit(":", 1)[-1]
        return KnowledgeEdge(
            id=f"wise_activity_csv:edge:{digest}",
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={"source_project": SourceProject.WISE_ACTIVITY_CSV.value, "relation_type": relation_type},
        )

    def _normalize_currency(self, value: Any) -> str:
        return str(value or "").strip().upper()

    def _currency_source_id(self, currency: str) -> str:
        return digest_source_id("wise_activity_csv:currency", currency)
