"""Adapter for Substack subscribers CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SubstackSubscribersCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "substack_subscribers_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["subscriber"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "subscriber" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None or (
                    sync_at
                    and (
                        not any(key in unit.metadata for key in ["subscribed_at", "created_at", "unsubscribed_at"])
                        or unit.updated_at <= sync_at
                    )
                ):
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        email = first(row, "Email", "Email Address", "Subscriber Email")
        name = first(row, "Name", "Subscriber Name")
        subscription_type = first(row, "Subscription Type", "Plan", "Tier")
        status = first(row, "Status", "Subscription Status")
        pledge = first(row, "Pledge", "Price", "Amount")
        source = first(row, "Source", "Signup Source")
        subscribed_at_text = first(row, "Subscribed At", "Subscription Started", "Created At", "Joined At")
        created_at_text = first(row, "Created At", "Joined At")
        unsubscribed_at_text = first(row, "Unsubscribed At", "Cancelled At")
        stripe_customer_id = first(row, "Stripe Customer ID", "Stripe Customer Id", "Customer ID")
        subscribed_at = parse_datetime(subscribed_at_text)
        created_at = parse_datetime(created_at_text)
        unsubscribed_at = parse_datetime(unsubscribed_at_text)
        if not any([email, name, subscription_type, status, pledge, source, subscribed_at_text, created_at_text, unsubscribed_at_text, stripe_customer_id]):
            return None
        now = datetime.now(timezone.utc)
        timestamp = subscribed_at or created_at or unsubscribed_at or now
        metadata = clean_metadata(
            {
                "email": email,
                "name": name,
                "subscription_type": subscription_type,
                "status": status,
                "pledge": pledge,
                "source": source,
                "subscribed_at": subscribed_at.isoformat() if subscribed_at else subscribed_at_text,
                "created_at": created_at.isoformat() if created_at else created_at_text,
                "unsubscribed_at": unsubscribed_at.isoformat() if unsubscribed_at else unsubscribed_at_text,
                "stripe_customer_id": stripe_customer_id,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="substack_subscribers_csv",
            source_id=f"substack_subscribers_csv:{email}" if email else digest_source_id("substack_subscribers_csv", name, subscription_type, status, source, index),
            source_entity_type="subscriber",
            title=name or email or "Substack subscriber",
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["substack", "subscriber", subscription_type, status, source] if tag)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _content(self, metadata: dict[str, Any]) -> str:
        labels = [("Name", "name"), ("Email", "email"), ("Subscription Type", "subscription_type"), ("Status", "status"), ("Pledge", "pledge"), ("Source", "source")]
        return "\n".join(f"{label}: {metadata[key]}" for label, key in labels if key in metadata)
