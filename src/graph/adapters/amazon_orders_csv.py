"""Adapter for Amazon retail order CSV exports."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class AmazonOrdersCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "amazon_orders_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["order", "item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or self.entity_types)
        if not allowed.intersection(self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        files: dict[str, set[str]] = defaultdict(set)
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                order_id = first(row, "Order ID", "order_id", "Order Id")
                if not order_id:
                    order_id = digest_source_id("order", first(row, "Order Date"), first(row, "Title"))
                groups[order_id].append(row)
                files[order_id].add(path.name)
        for order_id, rows in groups.items():
            order, items = self._units(order_id, rows, sorted(files[order_id]))
            if sync_at and order.updated_at <= sync_at:
                continue
            if "order" in allowed:
                result.units.append(order)
            if "item" in allowed:
                result.units.extend(items)
            if {"order", "item"}.issubset(allowed):
                for item in items:
                    result.edges.append(self._edge(order.source_id, item.source_id))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _units(self, order_id: str, rows: list[dict[str, Any]], source_files: list[str]) -> tuple[KnowledgeUnit, list[KnowledgeUnit]]:
        first_row = rows[0]
        order_date = parse_datetime(first(first_row, "Order Date", "order_date"))
        shipment_date = parse_datetime(first(first_row, "Shipment Date", "Ship Date"))
        categories: Counter[str] = Counter()
        total = 0.0
        items: list[KnowledgeUnit] = []
        now = datetime.now(timezone.utc)
        for index, row in enumerate(rows):
            category = first(row, "Category")
            if category:
                categories[category] += 1
            owed = parse_money(first(row, "Total Owed", "Item Total", "Total"))
            if owed is not None:
                total += owed
            item_metadata = {
                "order_id": order_id,
                "order_date": order_date.isoformat() if order_date else first(row, "Order Date"),
                "shipment_date": shipment_date.isoformat() if shipment_date else first(row, "Shipment Date"),
                "title": first(row, "Title", "Product Name"),
                "category": category,
                "asin": first(row, "ASIN", "ASIN/ISBN", "ISBN"),
                "quantity": parse_int(first(row, "Quantity")),
                "purchase_price_per_unit": parse_money(first(row, "Purchase Price Per Unit", "Unit Price")),
                "total_owed": owed,
                "seller": first(row, "Seller"),
                "condition": first(row, "Condition"),
                "product_url": first(row, "Product URL", "URL"),
                "position": index + 1,
            }
            title = item_metadata["title"] or f"Amazon item {index + 1}"
            items.append(
                KnowledgeUnit(
                    source_project=SourceProject.AMAZON_ORDERS_CSV,
                    source_id=digest_source_id("amazon_orders_csv:item", order_id, item_metadata.get("asin"), title, index),
                    source_entity_type="item",
                    title=str(title),
                    content=str(title),
                    content_type=ContentType.METADATA,
                    metadata=clean_metadata(item_metadata),
                    tags=list(dict.fromkeys(tag for tag in ["amazon", "item", category] if tag)),
                    created_at=order_date or now,
                    updated_at=shipment_date or order_date or now,
                )
            )
        title = f"Amazon order {order_id}"
        order = KnowledgeUnit(
            source_project=SourceProject.AMAZON_ORDERS_CSV,
            source_id=f"amazon_orders_csv:{order_id}",
            source_entity_type="order",
            title=title,
            content=f"{title}\nItems: {len(items)}",
            content_type=ContentType.METADATA,
            metadata=clean_metadata({"order_id": order_id, "order_date": order_date.isoformat() if order_date else first(first_row, "Order Date"), "shipment_date": shipment_date.isoformat() if shipment_date else first(first_row, "Shipment Date"), "item_count": len(items), "total_owed": round(total, 2) if total else None, "categories": dict(sorted(categories.items())), "source_files": source_files}),
            tags=list(dict.fromkeys(tag for tag in ["amazon", "order", *categories.keys()] if tag)),
            created_at=order_date or now,
            updated_at=shipment_date or order_date or now,
        )
        return order, items

    def _edge(self, order_id: str, item_id: str) -> KnowledgeEdge:
        return KnowledgeEdge(id=digest_source_id("amazon_orders_csv:contains", order_id, item_id), from_unit_id=order_id, to_unit_id=item_id, relation=EdgeRelation.CONTAINS, source=EdgeSource.SOURCE, metadata={"source_project": SourceProject.AMAZON_ORDERS_CSV.value})
