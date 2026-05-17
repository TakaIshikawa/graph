"""Adapter for PayPal activity CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class PaypalActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "paypal_activity_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["activity", "counterparty"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or ["activity"])
        if not allowed_types.intersection(self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        activities: list[KnowledgeUnit] = []

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
                activities.append(unit)
                if "activity" in allowed_types:
                    result.units.append(unit)

        counterparties = self._counterparty_units(activities) if "counterparty" in allowed_types else []
        if "counterparty" in allowed_types:
            result.units.extend(counterparties)
        if {"activity", "counterparty"}.issubset(allowed_types):
            result.edges.extend(self._counterparty_edges(activities, counterparties))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        transaction_id = first(row, "Transaction ID", "Transaction Id", "ID", "Txn ID")
        date_text = first(row, "Date", "Timestamp", "Time", "Datetime")
        timestamp = self._timestamp(row, date_text)
        name = first(row, "Name", "From Email Address", "To Email Address", "Counterparty")
        activity_type = first(row, "Type", "Transaction Type")
        status = first(row, "Status")
        note = first(row, "Subject", "Note", "Item Title", "Description")
        currency = first(row, "Currency", "Currency Code")
        gross = self._amount(first(row, "Gross", "Amount", "Total"))
        fee = self._amount(first(row, "Fee"))
        net = self._amount(first(row, "Net"))
        balance = self._amount(first(row, "Balance", "Running Balance"))
        reference = first(row, "Reference Txn ID", "Reference ID", "Invoice Number", "Receipt ID")
        if not any([transaction_id, date_text, name, activity_type, note, gross is not None, net is not None]):
            return None

        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat() if timestamp else date_text,
                "name": name,
                "type": activity_type,
                "status": status,
                "subject": first(row, "Subject"),
                "note": note,
                "gross": gross,
                "net": net,
                "fee": fee,
                "currency": currency,
                "balance": balance,
                "balance_impact": net if net is not None else gross,
                "reference": reference,
                "from_email": first(row, "From Email Address"),
                "to_email": first(row, "To Email Address"),
                "counterparty": name,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.PAYPAL_ACTIVITY_CSV,
            source_id=f"paypal_activity_csv:{transaction_id}" if transaction_id else digest_source_id("paypal_activity_csv", date_text, name, activity_type, gross, net, index),
            source_entity_type="activity",
            title=self._title(name, activity_type, gross, currency),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["paypal", "activity", activity_type, status, currency] if tag)),
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _timestamp(self, row: dict[str, Any], date_text: str) -> datetime | None:
        time_text = first(row, "Time")
        if date_text and time_text and time_text not in date_text:
            parsed = parse_datetime(f"{date_text} {time_text}")
            if parsed:
                return parsed
        return parse_datetime(date_text)

    def _amount(self, value: str) -> float | None:
        text = value.strip()
        if not text:
            return None
        negative = text.startswith("(") and text.endswith(")")
        cleaned = re.sub(r"[^0-9.\-]", "", text)
        if cleaned in {"", "-", "."}:
            return None
        try:
            amount = float(cleaned)
        except ValueError:
            return None
        return -abs(amount) if negative else amount

    def _title(self, name: str, activity_type: str, amount: float | None, currency: str) -> str:
        parts = [part for part in [activity_type, name] if part]
        if amount is not None:
            parts.append(f"{amount:g} {currency}".strip())
        return " - ".join(parts) or "PayPal activity"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            f"Name: {metadata.get('name')}" if metadata.get("name") else "",
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Amount: {metadata.get('gross')} {metadata.get('currency', '')}".strip() if metadata.get("gross") is not None else "",
            f"Fee: {metadata.get('fee')}" if metadata.get("fee") is not None else "",
            f"Net: {metadata.get('net')}" if metadata.get("net") is not None else "",
            f"Note: {metadata.get('note')}" if metadata.get("note") else "",
        ]
        return "\n".join(part for part in parts if part)

    def _counterparty_units(self, activities: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        labels: dict[str, str] = {}
        for activity in activities:
            counterparty = str(activity.metadata.get("counterparty") or "").strip()
            normalized = self._normalize_counterparty(counterparty)
            if not normalized:
                continue
            grouped.setdefault(normalized, []).append(activity)
            labels.setdefault(normalized, counterparty)

        units: list[KnowledgeUnit] = []
        for normalized, items in sorted(grouped.items()):
            totals = [float(item.metadata.get("balance_impact")) for item in items if isinstance(item.metadata.get("balance_impact"), int | float)]
            currencies = sorted({str(item.metadata.get("currency")) for item in items if item.metadata.get("currency")})
            statuses = sorted({str(item.metadata.get("status")) for item in items if item.metadata.get("status")})
            metadata = clean_metadata(
                {
                    "counterparty": labels[normalized],
                    "normalized_counterparty": normalized,
                    "activity_count": len(items),
                    "first_seen": min(item.created_at for item in items).isoformat(),
                    "last_seen": max(item.created_at for item in items).isoformat(),
                    "net_total": sum(totals) if totals else None,
                    "currencies": currencies,
                    "statuses": statuses,
                    "activity_source_ids": sorted(item.source_id for item in items),
                }
            )
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.PAYPAL_ACTIVITY_CSV,
                    source_id=self._counterparty_source_id(normalized),
                    source_entity_type="counterparty",
                    title=labels[normalized],
                    content=f"PayPal counterparty: {labels[normalized]}\nActivities: {len(items)}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=["paypal", "counterparty", labels[normalized]],
                    created_at=min(item.created_at for item in items),
                    updated_at=max(item.updated_at for item in items),
                )
            )
        return units

    def _counterparty_edges(self, activities: list[KnowledgeUnit], counterparties: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        counterparty_ids = {unit.metadata["normalized_counterparty"]: unit.source_id for unit in counterparties}
        edges = []
        for activity in activities:
            target = counterparty_ids.get(self._normalize_counterparty(activity.metadata.get("counterparty")))
            if target:
                edges.append(self._edge(activity.source_id, target, "activity_counterparty"))
        return list({edge.id: edge for edge in edges}.values())

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        digest = digest_source_id("paypal_activity_csv:edge", from_id, relation_type, to_id).rsplit(":", 1)[-1]
        return KnowledgeEdge(
            id=f"paypal_activity_csv:edge:{digest}",
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={"source_project": SourceProject.PAYPAL_ACTIVITY_CSV.value, "relation_type": relation_type},
        )

    def _normalize_counterparty(self, value: Any) -> str:
        return " ".join(str(value or "").strip().casefold().split())

    def _counterparty_source_id(self, normalized: str) -> str:
        return digest_source_id("paypal_activity_csv:counterparty", normalized)
