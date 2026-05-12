"""Adapter for Instacart order CSV exports."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class InstacartOrdersCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "instacart_orders_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["order"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types and "order" not in entity_types:
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
                order_id = first(row, "order_id", "Order ID", "Order")
                if not order_id:
                    order_id = digest_source_id("row", first(row, "ordered_at", "Ordered At"), first(row, "store_name", "Store Name"))
                groups[order_id].append(row)
                files[order_id].add(path.name)
        for order_id, rows in groups.items():
            unit = self._unit(order_id, rows, sorted(files[order_id]))
            if sync_at and unit.updated_at <= sync_at:
                continue
            result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, order_id: str, rows: list[dict[str, Any]], source_files: list[str]) -> KnowledgeUnit:
        first_row = rows[0]
        ordered_at = parse_datetime(first(first_row, "ordered_at", "Ordered At", "Order Date"))
        store = first(first_row, "store_name", "Store Name", "Store")
        items = []
        departments: Counter[str] = Counter()
        total = 0.0
        for index, row in enumerate(rows):
            item = {
                "name": first(row, "item_name", "Item Name", "Item"),
                "quantity": parse_int(first(row, "quantity", "Quantity")),
                "unit_price": parse_money(first(row, "unit_price", "Unit Price")),
                "total_price": parse_money(first(row, "total_price", "Total Price", "Item Total")),
                "aisle": first(row, "aisle", "Aisle"),
                "department": first(row, "department", "Department"),
                "replacement": first(row, "replacement", "Replacement"),
                "item_url": first(row, "item_url", "Item URL", "URL"),
                "position": index + 1,
            }
            if item["department"]:
                departments[str(item["department"])] += 1
            if item["total_price"] is not None:
                total += float(item["total_price"])
            items.append(clean_metadata(item))
        metadata = {
            "order_id": order_id,
            "ordered_at": ordered_at.isoformat() if ordered_at else first(first_row, "ordered_at", "Ordered At"),
            "store_name": store,
            "item_count": len(items),
            "total_price": round(total, 2) if total else None,
            "departments": dict(sorted(departments.items())),
            "items": sorted(items, key=lambda item: (str(item.get("name", "")), int(item.get("position", 0)))),
            "source_files": source_files,
        }
        now = datetime.now(timezone.utc)
        title = f"Instacart order {order_id}" if not store else f"Instacart order from {store}"
        return KnowledgeUnit(
            source_project=SourceProject.INSTACART_ORDERS_CSV,
            source_id=f"instacart_orders_csv:{order_id}",
            source_entity_type="order",
            title=title,
            content=f"{title}\nItems: {len(items)}",
            content_type=ContentType.METADATA,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["instacart", "order", store, *departments.keys()] if tag)),
            created_at=ordered_at or now,
            updated_at=ordered_at or now,
        )
