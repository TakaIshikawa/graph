"""Adapter for Coinbase transaction history CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class CoinbaseTransactionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "coinbase_transactions_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["transaction", "asset"]

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

        assets = self._asset_units(transactions) if "asset" in allowed_types else []
        if "asset" in allowed_types:
            result.units.extend(assets)
        if {"transaction", "asset"}.issubset(allowed_types):
            result.edges.extend(self._asset_edges(transactions, assets))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID", "Order ID")
        timestamp_text = first(row, "Timestamp", "Date", "Time", "Created At")
        timestamp = parse_datetime(timestamp_text)
        transaction_type = first(row, "Transaction Type", "Type")
        asset = first(row, "Asset", "Asset Symbol", "Currency", "Currency/BTC")
        quantity = parse_float(first(row, "Quantity Transacted", "Quantity", "Amount", "Asset Amount"))
        spot_price = parse_float(first(row, "Spot Price at Transaction", "Spot Price", "Price"))
        subtotal = parse_float(first(row, "Subtotal", "Subtotal Amount"))
        total = parse_float(first(row, "Total (inclusive of fees and/or spread)", "Total", "Total Amount"))
        fees = parse_float(first(row, "Fees and/or Spread", "Fee", "Fees"))
        notes = first(row, "Notes", "Description", "Details")
        currency = first(row, "Subtotal Currency", "Total Currency", "Fiat Currency", "Currency Code")
        if not any([transaction_id, timestamp_text, transaction_type, asset, quantity is not None, total is not None, notes]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else timestamp_text,
                "transaction_type": transaction_type,
                "asset": asset,
                "quantity": quantity,
                "spot_price": spot_price,
                "subtotal": subtotal,
                "total": total,
                "fees": fees,
                "currency": currency,
                "notes": notes,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.COINBASE_TRANSACTIONS_CSV,
            source_id=f"coinbase_transactions_csv:{transaction_id}" if transaction_id else digest_source_id("coinbase_transactions_csv", timestamp_text, transaction_type, asset, quantity, total, index),
            source_entity_type="transaction",
            title=self._title(transaction_type, asset, quantity),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["coinbase", "crypto", asset, transaction_type] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _title(self, transaction_type: str, asset: str, quantity: float | None) -> str:
        title = " ".join(part for part in [transaction_type, asset] if part) or "Coinbase transaction"
        if quantity is not None:
            return f"{title} ({quantity:g})"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Type: {metadata.get('transaction_type')}" if metadata.get("transaction_type") else "",
            f"Asset: {metadata.get('asset')}" if metadata.get("asset") else "",
            f"Quantity: {metadata.get('quantity')}" if metadata.get("quantity") is not None else "",
            f"Spot price: {metadata.get('spot_price')}" if metadata.get("spot_price") is not None else "",
            f"Total: {metadata.get('total')} {metadata.get('currency', '')}".strip() if metadata.get("total") is not None else "",
            f"Fees: {metadata.get('fees')}" if metadata.get("fees") is not None else "",
            f"Notes: {metadata.get('notes')}" if metadata.get("notes") else "",
        ]
        return "\n".join(part for part in parts if part)

    def _asset_units(self, transactions: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        labels: dict[str, str] = {}
        for transaction in transactions:
            asset = str(transaction.metadata.get("asset") or "").strip()
            normalized = self._normalize_asset(asset)
            if not normalized:
                continue
            grouped.setdefault(normalized, []).append(transaction)
            labels.setdefault(normalized, asset.upper() if asset else normalized)

        units: list[KnowledgeUnit] = []
        for normalized, items in sorted(grouped.items()):
            quantities = [float(item.metadata["quantity"]) for item in items if isinstance(item.metadata.get("quantity"), int | float)]
            native_totals = [float(item.metadata["total"]) for item in items if isinstance(item.metadata.get("total"), int | float)]
            types = sorted({str(item.metadata.get("transaction_type")) for item in items if item.metadata.get("transaction_type")})
            currencies = sorted({str(item.metadata.get("currency")) for item in items if item.metadata.get("currency")})
            metadata = clean_metadata(
                {
                    "asset": labels[normalized],
                    "normalized_asset": normalized,
                    "transaction_count": len(items),
                    "total_quantity": sum(quantities) if quantities else None,
                    "native_amount_total": sum(native_totals) if native_totals else None,
                    "native_currencies": currencies,
                    "first_seen": min(item.created_at for item in items).isoformat(),
                    "last_seen": max(item.created_at for item in items).isoformat(),
                    "transaction_types": types,
                    "transaction_source_ids": sorted(item.source_id for item in items),
                }
            )
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.COINBASE_TRANSACTIONS_CSV,
                    source_id=self._asset_source_id(normalized),
                    source_entity_type="asset",
                    title=labels[normalized],
                    content=f"Coinbase asset: {labels[normalized]}\nTransactions: {len(items)}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=["coinbase", "asset", labels[normalized]],
                    created_at=min(item.created_at for item in items),
                    updated_at=max(item.updated_at for item in items),
                )
            )
        return units

    def _asset_edges(self, transactions: list[KnowledgeUnit], assets: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        asset_ids = {unit.metadata["normalized_asset"]: unit.source_id for unit in assets}
        edges = []
        for transaction in transactions:
            target = asset_ids.get(self._normalize_asset(transaction.metadata.get("asset")))
            if target:
                edges.append(self._edge(transaction.source_id, target, "transaction_asset"))
        return list({edge.id: edge for edge in edges}.values())

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        digest = digest_source_id("coinbase_transactions_csv:edge", from_id, relation_type, to_id).rsplit(":", 1)[-1]
        return KnowledgeEdge(
            id=f"coinbase_transactions_csv:edge:{digest}",
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={"source_project": SourceProject.COINBASE_TRANSACTIONS_CSV.value, "relation_type": relation_type},
        )

    def _normalize_asset(self, value: Any) -> str:
        return str(value or "").strip().upper()

    def _asset_source_id(self, normalized: str) -> str:
        return digest_source_id("coinbase_transactions_csv:asset", normalized)
