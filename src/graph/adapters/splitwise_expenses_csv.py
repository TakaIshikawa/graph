"""Adapter for Splitwise expenses CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class SplitwiseExpensesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "splitwise_expenses_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["expense", "group"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or ["expense"])
        if not allowed_types.intersection(self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        expenses: list[KnowledgeUnit] = []

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
                expenses.append(unit)
                if "expense" in allowed_types:
                    result.units.append(unit)

        groups = self._group_units(expenses) if "group" in allowed_types else []
        if "group" in allowed_types:
            result.units.extend(groups)
        if {"expense", "group"}.issubset(allowed_types):
            result.edges.extend(self._group_edges(expenses, groups))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        expense_id = first(row, "Expense ID", "Expense Id", "ID")
        date_text = first(row, "Date", "Created At", "Created")
        updated_text = first(row, "Updated At", "Updated", "Date")
        timestamp = parse_datetime(date_text)
        updated_at = parse_datetime(updated_text) or timestamp
        description = first(row, "Description", "Expense", "Details")
        category = first(row, "Category")
        cost = parse_float(first(row, "Cost", "Amount", "Total"))
        currency = first(row, "Currency", "Currency Code")
        group = first(row, "Group", "Group Name")
        paid_by = first(row, "Paid By", "Payer")
        owed_by = first(row, "Owed By", "Participants", "Split Between")
        users = split_values(first(row, "Users", "Members"))
        comments = first(row, "Comments", "Comment", "Notes")
        settled = first(row, "Settled", "Is Settled")
        if not any([expense_id, date_text, description, category, cost is not None, group, paid_by, owed_by, comments]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "expense_id": expense_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "updated_at": updated_at.isoformat() if updated_at else updated_text,
                "description": description,
                "category": category,
                "cost": cost,
                "currency": currency,
                "group": group,
                "paid_by": paid_by,
                "owed_by": owed_by,
                "users": users,
                "comments": comments,
                "settled": settled,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.SPLITWISE_EXPENSES_CSV,
            source_id=f"splitwise_expenses_csv:{expense_id}" if expense_id else digest_source_id("splitwise_expenses_csv", date_text, description, cost, paid_by, owed_by, index),
            source_entity_type="expense",
            title=self._title(description, cost, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["splitwise", "expense", category, currency] if tag)),
            created_at=timestamp or now,
            updated_at=updated_at or timestamp or now,
        )

    def _title(self, description: str, cost: float | None, currency: str) -> str:
        if cost is not None:
            return f"{description or 'Splitwise expense'} ({cost:g} {currency})".strip()
        return description or "Splitwise expense"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            metadata.get("description", ""),
            f"Group: {metadata.get('group')}" if metadata.get("group") else "",
            f"Category: {metadata.get('category')}" if metadata.get("category") else "",
            f"Paid by: {metadata.get('paid_by')}" if metadata.get("paid_by") else "",
            f"Owed by: {metadata.get('owed_by')}" if metadata.get("owed_by") else "",
            f"Users: {', '.join(metadata.get('users', []))}" if metadata.get("users") else "",
            f"Comments: {metadata.get('comments')}" if metadata.get("comments") else "",
        ]
        return "\n".join(part for part in parts if part)

    def _group_units(self, expenses: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        labels: dict[str, str] = {}
        for expense in expenses:
            group = str(expense.metadata.get("group") or "").strip()
            normalized = self._normalize_group(group)
            if not normalized:
                continue
            grouped.setdefault(normalized, []).append(expense)
            labels.setdefault(normalized, group)

        units: list[KnowledgeUnit] = []
        for normalized, items in sorted(grouped.items()):
            costs = [float(item.metadata["cost"]) for item in items if isinstance(item.metadata.get("cost"), int | float)]
            participants = sorted(
                {person for item in items for person in [str(item.metadata.get("paid_by") or "").strip(), str(item.metadata.get("owed_by") or "").strip(), *item.metadata.get("users", [])] if person}
            )
            categories = sorted({str(item.metadata.get("category")) for item in items if item.metadata.get("category")})
            currencies = sorted({str(item.metadata.get("currency")) for item in items if item.metadata.get("currency")})
            metadata = clean_metadata(
                {
                    "group": labels[normalized],
                    "normalized_group": normalized,
                    "expense_count": len(items),
                    "participants": participants,
                    "categories": categories,
                    "total_cost": sum(costs) if costs else None,
                    "currencies": currencies,
                    "first_seen": min(item.created_at for item in items).isoformat(),
                    "last_seen": max(item.created_at for item in items).isoformat(),
                    "expense_source_ids": sorted(item.source_id for item in items),
                }
            )
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.SPLITWISE_EXPENSES_CSV,
                    source_id=self._group_source_id(normalized),
                    source_entity_type="group",
                    title=labels[normalized],
                    content=f"Splitwise group: {labels[normalized]}\nExpenses: {len(items)}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=["splitwise", "group", labels[normalized]],
                    created_at=min(item.created_at for item in items),
                    updated_at=max(item.updated_at for item in items),
                )
            )
        return units

    def _group_edges(self, expenses: list[KnowledgeUnit], groups: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        group_ids = {unit.metadata["normalized_group"]: unit.source_id for unit in groups}
        edges = []
        for expense in expenses:
            target = group_ids.get(self._normalize_group(expense.metadata.get("group")))
            if target:
                edges.append(self._edge(expense.source_id, target, "expense_group"))
        return list({edge.id: edge for edge in edges}.values())

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        digest = digest_source_id("splitwise_expenses_csv:edge", from_id, relation_type, to_id).rsplit(":", 1)[-1]
        return KnowledgeEdge(
            id=f"splitwise_expenses_csv:edge:{digest}",
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={"source_project": SourceProject.SPLITWISE_EXPENSES_CSV.value, "relation_type": relation_type},
        )

    def _normalize_group(self, value: Any) -> str:
        return " ".join(str(value or "").strip().casefold().split())

    def _group_source_id(self, normalized: str) -> str:
        return digest_source_id("splitwise_expenses_csv:group", normalized)
