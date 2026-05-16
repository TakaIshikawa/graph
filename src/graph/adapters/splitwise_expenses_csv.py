"""Adapter for Splitwise expenses CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class SplitwiseExpensesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "splitwise_expenses_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["expense"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "expense" not in entity_types:
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
