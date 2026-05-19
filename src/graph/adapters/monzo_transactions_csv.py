"""Adapter for Monzo transaction CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class MonzoTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "monzo_transactions_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["transaction", "merchant"]

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

        merchants = self._merchant_units(transactions) if "merchant" in allowed_types else []
        if "merchant" in allowed_types:
            result.units.extend(merchants)
        if {"transaction", "merchant"}.issubset(allowed_types):
            result.edges.extend(self._merchant_edges(transactions, merchants))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
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

    def _merchant_units(self, transactions: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        labels: dict[str, str] = {}
        for transaction in transactions:
            merchant = str(transaction.metadata.get("merchant") or "").strip()
            normalized = self._normalize_merchant(merchant)
            if not normalized:
                continue
            grouped.setdefault(normalized, []).append(transaction)
            labels.setdefault(normalized, merchant)

        units: list[KnowledgeUnit] = []
        for normalized, items in sorted(grouped.items()):
            amounts = [float(item.metadata["amount"]) for item in items if isinstance(item.metadata.get("amount"), int | float)]
            currencies = sorted({str(item.metadata.get("currency")) for item in items if item.metadata.get("currency")})
            metadata = clean_metadata(
                {
                    "merchant": labels[normalized],
                    "normalized_merchant": normalized,
                    "transaction_count": len(items),
                    "first_seen": min(item.created_at for item in items).isoformat(),
                    "last_seen": max(item.created_at for item in items).isoformat(),
                    "total_amount": sum(amounts) if amounts else None,
                    "currency": currencies[0] if len(currencies) == 1 else None,
                    "currencies": currencies if len(currencies) > 1 else None,
                    "transaction_source_ids": sorted(item.source_id for item in items),
                }
            )
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.MONZO_TRANSACTIONS_CSV,
                    source_id=self._merchant_source_id(normalized),
                    source_entity_type="merchant",
                    title=labels[normalized],
                    content=f"Monzo merchant: {labels[normalized]}\nTransactions: {len(items)}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=["monzo", "merchant", labels[normalized]],
                    created_at=min(item.created_at for item in items),
                    updated_at=max(item.updated_at for item in items),
                )
            )
        return units

    def _merchant_edges(self, transactions: list[KnowledgeUnit], merchants: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        merchant_ids = {merchant.metadata["normalized_merchant"]: merchant.source_id for merchant in merchants}
        edges: list[KnowledgeEdge] = []
        for transaction in transactions:
            normalized = self._normalize_merchant(transaction.metadata.get("merchant"))
            merchant_id = merchant_ids.get(normalized)
            if merchant_id:
                edges.append(self._edge(transaction.source_id, merchant_id, "transaction_merchant"))
        return list({edge.id: edge for edge in edges}.values())

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        digest = digest_source_id("monzo_transactions_csv:edge", from_id, relation_type, to_id).rsplit(":", 1)[-1]
        return KnowledgeEdge(
            id=f"monzo_transactions_csv:edge:{digest}",
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={"source_project": SourceProject.MONZO_TRANSACTIONS_CSV.value, "relation_type": relation_type},
        )

    def _normalize_merchant(self, value: Any) -> str:
        return " ".join(str(value or "").strip().casefold().split())

    def _merchant_source_id(self, normalized: str) -> str:
        return digest_source_id("monzo_transactions_csv:merchant", normalized)
