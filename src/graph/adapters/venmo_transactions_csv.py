"""Adapter for Venmo transaction CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class VenmoTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "venmo_transactions_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["transaction", "counterparty"]

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

        counterparties = self._counterparty_units(transactions) if "counterparty" in allowed_types else []
        if "counterparty" in allowed_types:
            result.units.extend(counterparties)
        if {"transaction", "counterparty"}.issubset(allowed_types):
            result.edges.extend(self._counterparty_edges(transactions, counterparties))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID")
        date_text = first(row, "Datetime", "Date", "Completed Date", "Timestamp")
        timestamp = parse_datetime(date_text)
        transaction_type = first(row, "Type", "Transaction Type")
        status = first(row, "Status")
        note = first(row, "Note", "Description", "Memo")
        from_person = first(row, "From", "From User")
        to_person = first(row, "To", "To User")
        amount = self._amount(first(row, "Amount", "Total"))
        currency = first(row, "Currency")
        fee = self._amount(first(row, "Fee"))
        funding_source = first(row, "Funding Source", "Funding")
        destination = first(row, "Destination", "Bank/Card", "Bank")
        privacy = first(row, "Privacy", "Audience")
        url = first(row, "URL", "Transaction URL", "Link")
        if not any([transaction_id, date_text, transaction_type, note, from_person, to_person, amount is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "type": transaction_type,
                "status": status,
                "note": note,
                "from": from_person,
                "to": to_person,
                "counterparty": self._counterparty(from_person, to_person, amount),
                "amount": amount,
                "currency": currency,
                "fee": fee,
                "funding_source": funding_source,
                "destination": destination,
                "privacy": privacy,
                "url": url,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.VENMO_TRANSACTIONS_CSV,
            source_id=f"venmo_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("venmo_transactions_csv", date_text, transaction_type, note, from_person, to_person, amount, index),
            source_entity_type="transaction",
            title=self._title(transaction_type, note, amount),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["venmo", "transaction", transaction_type, status, privacy] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _amount(self, value: str) -> float | None:
        text = value.strip()
        if not text:
            return None
        negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
        cleaned = re.sub(r"[^0-9.]", "", text)
        if not cleaned:
            return None
        try:
            amount = float(cleaned)
        except ValueError:
            return None
        return -abs(amount) if negative else amount

    def _title(self, transaction_type: str, note: str, amount: float | None) -> str:
        parts = [part for part in [transaction_type, note] if part]
        if amount is not None:
            parts.append(f"{amount:g}")
        return " - ".join(parts) or "Venmo transaction"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Note: {metadata.get('note')}" if metadata.get("note") else "",
            f"From: {metadata.get('from')}" if metadata.get("from") else "",
            f"To: {metadata.get('to')}" if metadata.get("to") else "",
            f"Amount: {metadata.get('amount')}" if metadata.get("amount") is not None else "",
            f"Currency: {metadata.get('currency')}" if metadata.get("currency") else "",
            f"Fee: {metadata.get('fee')}" if metadata.get("fee") is not None else "",
            f"URL: {metadata.get('url')}" if metadata.get("url") else "",
        ]
        return "\n".join(part for part in parts if part)

    def _counterparty(self, from_person: str, to_person: str, amount: float | None) -> str:
        if from_person and to_person:
            return to_person if amount is not None and amount < 0 else from_person
        return from_person or to_person

    def _counterparty_units(self, transactions: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        labels: dict[str, str] = {}
        for transaction in transactions:
            counterparty = str(transaction.metadata.get("counterparty") or "").strip()
            normalized = self._normalize_counterparty(counterparty)
            if not normalized:
                continue
            grouped.setdefault(normalized, []).append(transaction)
            labels.setdefault(normalized, counterparty)

        units: list[KnowledgeUnit] = []
        for normalized, items in sorted(grouped.items()):
            amounts = [float(item.metadata["amount"]) for item in items if isinstance(item.metadata.get("amount"), int | float)]
            types = sorted({str(item.metadata.get("type")) for item in items if item.metadata.get("type")})
            metadata = clean_metadata(
                {
                    "counterparty": labels[normalized],
                    "normalized_counterparty": normalized,
                    "transaction_count": len(items),
                    "first_seen": min(item.created_at for item in items).isoformat(),
                    "last_seen": max(item.created_at for item in items).isoformat(),
                    "net_amount": sum(amounts) if amounts else None,
                    "transaction_types": types,
                    "transaction_source_ids": sorted(item.source_id for item in items),
                }
            )
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.VENMO_TRANSACTIONS_CSV,
                    source_id=self._counterparty_source_id(normalized),
                    source_entity_type="counterparty",
                    title=labels[normalized],
                    content=f"Venmo counterparty: {labels[normalized]}\nTransactions: {len(items)}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=["venmo", "counterparty", labels[normalized]],
                    created_at=min(item.created_at for item in items),
                    updated_at=max(item.updated_at for item in items),
                )
            )
        return units

    def _counterparty_edges(self, transactions: list[KnowledgeUnit], counterparties: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        counterparty_ids = {unit.metadata["normalized_counterparty"]: unit.source_id for unit in counterparties}
        edges = []
        for transaction in transactions:
            target = counterparty_ids.get(self._normalize_counterparty(transaction.metadata.get("counterparty")))
            if target:
                edges.append(self._edge(transaction.source_id, target, "transaction_counterparty"))
        return list({edge.id: edge for edge in edges}.values())

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        digest = digest_source_id("venmo_transactions_csv:edge", from_id, relation_type, to_id).rsplit(":", 1)[-1]
        return KnowledgeEdge(
            id=f"venmo_transactions_csv:edge:{digest}",
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={"source_project": SourceProject.VENMO_TRANSACTIONS_CSV.value, "relation_type": relation_type},
        )

    def _normalize_counterparty(self, value: Any) -> str:
        return " ".join(str(value or "").strip().casefold().split())

    def _counterparty_source_id(self, normalized: str) -> str:
        return digest_source_id("venmo_transactions_csv:counterparty", normalized)
