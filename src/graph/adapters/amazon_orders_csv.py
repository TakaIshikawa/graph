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
        return ["order", "shipment", "item", "return"]

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
            order, shipments, items, returns = self._units(order_id, rows, sorted(files[order_id]))
            if sync_at and order.updated_at <= sync_at:
                continue
            if "order" in allowed:
                result.units.append(order)
            if "shipment" in allowed:
                result.units.extend(shipments)
            if "item" in allowed:
                result.units.extend(items)
            if "return" in allowed:
                result.units.extend(returns)
            if {"order", "shipment"}.issubset(allowed):
                for shipment in shipments:
                    result.edges.append(self._edge(order.source_id, shipment.source_id))
            if {"shipment", "item"}.issubset(allowed):
                shipment_ids = {shipment.source_id for shipment in shipments}
                for item in items:
                    shipment_id = item.metadata.get("shipment_source_id")
                    if shipment_id in shipment_ids:
                        result.edges.append(self._edge(str(shipment_id), item.source_id))
            if {"order", "item"}.issubset(allowed):
                for item in items:
                    if not item.metadata.get("shipment_source_id"):
                        result.edges.append(self._edge(order.source_id, item.source_id))
            if "return" in allowed:
                if "order" in allowed:
                    for return_unit in returns:
                        result.edges.append(self._edge(order.source_id, return_unit.source_id))
                if "item" in allowed:
                    item_ids = {item.source_id for item in items}
                    for return_unit in returns:
                        item_source_id = return_unit.metadata.get("item_source_id")
                        if item_source_id in item_ids:
                            result.edges.append(self._edge(str(item_source_id), return_unit.source_id))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _units(
        self, order_id: str, rows: list[dict[str, Any]], source_files: list[str]
    ) -> tuple[KnowledgeUnit, list[KnowledgeUnit], list[KnowledgeUnit], list[KnowledgeUnit]]:
        first_row = rows[0]
        order_date = parse_datetime(first(first_row, "Order Date", "order_date"))
        shipment_date = parse_datetime(first(first_row, "Shipment Date", "Ship Date"))
        categories: Counter[str] = Counter()
        total = 0.0
        items: list[KnowledgeUnit] = []
        returns: list[KnowledgeUnit] = []
        shipment_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        shipment_item_counts: Counter[str] = Counter()
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
            shipment_key = self._shipment_key(order_id, row)
            if shipment_key:
                shipment_source_id = digest_source_id("amazon_orders_csv:shipment", order_id, shipment_key)
                item_metadata["shipment_source_id"] = shipment_source_id
                item_metadata["shipment_id"] = first(row, "Shipment ID", "Shipment Id", "shipment_id")
                item_metadata["tracking_number"] = first(row, "Tracking Number", "Tracking")
                item_metadata["carrier"] = first(row, "Carrier", "Shipping Carrier")
                shipment_rows[shipment_key].append(row)
                shipment_item_counts[shipment_key] += 1
            title = item_metadata["title"] or f"Amazon item {index + 1}"
            item_source_id = digest_source_id("amazon_orders_csv:item", order_id, item_metadata.get("asin"), title, index)
            item_unit = KnowledgeUnit(
                    source_project=SourceProject.AMAZON_ORDERS_CSV,
                    source_id=item_source_id,
                    source_entity_type="item",
                    title=str(title),
                    content=str(title),
                    content_type=ContentType.METADATA,
                    metadata=clean_metadata(item_metadata),
                    tags=list(dict.fromkeys(tag for tag in ["amazon", "item", category] if tag)),
                    created_at=order_date or now,
                    updated_at=shipment_date or order_date or now,
                )
            items.append(item_unit)
            return_unit = self._return_unit(order_id, row, index, item_unit, order_date, now, source_files)
            if return_unit is not None:
                returns.append(return_unit)
        shipments = [
            self._shipment_unit(order_id, key, grouped_rows, shipment_item_counts[key], source_files, order_date, now)
            for key, grouped_rows in sorted(shipment_rows.items())
        ]
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
        return order, shipments, items, returns

    def _return_unit(
        self,
        order_id: str,
        row: dict[str, Any],
        index: int,
        item_unit: KnowledgeUnit,
        order_date: datetime | None,
        now: datetime,
        source_files: list[str],
    ) -> KnowledgeUnit | None:
        return_date_text = first(row, "Return Date", "Returned Date", "Refund Date")
        reason = first(row, "Return Reason", "Refund Reason")
        status = first(row, "Return Status", "Refund Status")
        refund_amount = parse_money(first(row, "Refund Amount", "Refunded Amount", "Refund"))
        if not any([return_date_text, reason, status, refund_amount is not None]):
            return None
        return_date = parse_datetime(return_date_text)
        title = f"Amazon return for {item_unit.title}"
        metadata = clean_metadata(
            {
                "order_id": order_id,
                "item_source_id": item_unit.source_id,
                "item_title": item_unit.title,
                "asin": item_unit.metadata.get("asin"),
                "position": index + 1,
                "return_date": return_date.isoformat() if return_date else return_date_text,
                "return_reason": reason,
                "return_status": status,
                "refund_amount": refund_amount,
                "source_files": source_files,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.AMAZON_ORDERS_CSV,
            source_id=digest_source_id("amazon_orders_csv:return", order_id, item_unit.source_id, return_date_text, reason, status, refund_amount),
            source_entity_type="return",
            title=title,
            content=title,
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["amazon", "return"],
            created_at=return_date or order_date or now,
            updated_at=return_date or order_date or now,
        )

    def _shipment_key(self, order_id: str, row: dict[str, Any]) -> str:
        explicit = first(row, "Shipment ID", "Shipment Id", "shipment_id")
        if explicit:
            return explicit
        tracking = first(row, "Tracking Number", "Tracking")
        carrier = first(row, "Carrier", "Shipping Carrier")
        shipment_date = first(row, "Shipment Date", "Ship Date")
        if tracking:
            return f"tracking:{tracking}"
        if carrier or shipment_date:
            return "|".join([order_id, shipment_date, carrier])
        return ""

    def _shipment_unit(
        self,
        order_id: str,
        shipment_key: str,
        rows: list[dict[str, Any]],
        item_count: int,
        source_files: list[str],
        order_date: datetime | None,
        now: datetime,
    ) -> KnowledgeUnit:
        first_row = rows[0]
        shipment_date = parse_datetime(first(first_row, "Shipment Date", "Ship Date"))
        shipment_id = first(first_row, "Shipment ID", "Shipment Id", "shipment_id") or shipment_key
        carrier = first(first_row, "Carrier", "Shipping Carrier")
        tracking_number = first(first_row, "Tracking Number", "Tracking")
        title = f"Amazon shipment {shipment_id}"
        return KnowledgeUnit(
            source_project=SourceProject.AMAZON_ORDERS_CSV,
            source_id=digest_source_id("amazon_orders_csv:shipment", order_id, shipment_key),
            source_entity_type="shipment",
            title=title,
            content=f"{title}\nItems: {item_count}",
            content_type=ContentType.METADATA,
            metadata=clean_metadata(
                {
                    "order_id": order_id,
                    "shipment_id": shipment_id,
                    "shipment_date": shipment_date.isoformat() if shipment_date else first(first_row, "Shipment Date", "Ship Date"),
                    "carrier": carrier,
                    "tracking_number": tracking_number,
                    "item_count": item_count,
                    "source_files": source_files,
                }
            ),
            tags=list(dict.fromkeys(tag for tag in ["amazon", "shipment", carrier] if tag)),
            created_at=shipment_date or order_date or now,
            updated_at=shipment_date or order_date or now,
        )

    def _edge(self, order_id: str, item_id: str) -> KnowledgeEdge:
        return KnowledgeEdge(id=digest_source_id("amazon_orders_csv:contains", order_id, item_id), from_unit_id=order_id, to_unit_id=item_id, relation=EdgeRelation.CONTAINS, source=EdgeSource.SOURCE, metadata={"source_project": SourceProject.AMAZON_ORDERS_CSV.value})
