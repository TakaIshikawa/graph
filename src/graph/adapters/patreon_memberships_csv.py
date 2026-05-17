"""Adapter for Patreon membership CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PatreonMembershipsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "patreon_memberships_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["membership"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "membership" not in entity_types:
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
        member_id = first(row, "Member ID", "Member Id", "Patron ID", "Patron Id", "User ID", "ID")
        full_name = first(row, "Full Name", "Name", "Patron Name", "Member Name")
        email = first(row, "Email", "Email Address", "Patron Email")
        tier = first(row, "Tier", "Tier Title", "Membership Tier", "Reward")
        patron_status = first(row, "Patron Status", "Status", "Membership Status")
        pledge_amount = parse_money(first(row, "Pledge Amount", "Current Pledge", "Amount", "Pledge"))
        lifetime_amount = parse_money(first(row, "Lifetime Amount", "Lifetime Support", "Lifetime Value", "Total Pledged"))
        currency = first(row, "Currency", "Pledge Currency", "ISO Currency Code")
        join_text = first(row, "Join Date", "Joined At", "Patron Since", "Created At")
        last_charge_text = first(row, "Last Charge Date", "Last Charged At", "Last Payment Date")
        next_charge_text = first(row, "Next Charge Date", "Next Charged At", "Next Payment Date")
        joined_at = parse_datetime(join_text)
        last_charge_at = parse_datetime(last_charge_text)
        next_charge_at = parse_datetime(next_charge_text)
        last_charge_status = first(row, "Last Charge Status", "Charge Status", "Last Payment Status")
        address_country = first(row, "Address Country", "Country", "Shipping Country")
        note = first(row, "Note", "Notes", "Patron Note")

        if not any([member_id, full_name, email, tier, patron_status, pledge_amount is not None, lifetime_amount is not None, join_text, last_charge_text, last_charge_status, address_country, note]):
            return None

        now = datetime.now(timezone.utc)
        updated_at = last_charge_at or joined_at or next_charge_at or now
        metadata = clean_metadata(
            {
                "member_id": member_id,
                "full_name": full_name,
                "email": email,
                "tier": tier,
                "patron_status": patron_status,
                "pledge_amount": pledge_amount,
                "lifetime_amount": lifetime_amount,
                "currency": currency,
                "join_date": joined_at.isoformat() if joined_at else join_text,
                "last_charge_date": last_charge_at.isoformat() if last_charge_at else last_charge_text,
                "last_charge_status": last_charge_status,
                "next_charge_date": next_charge_at.isoformat() if next_charge_at else next_charge_text,
                "address_country": address_country,
                "note": note,
                "source_file": source_file,
            }
        )

        return KnowledgeUnit(
            source_project="patreon_memberships_csv",
            source_id=f"patreon_memberships_csv:{member_id}" if member_id else digest_source_id("patreon_memberships_csv", full_name, email, tier, patron_status, pledge_amount, join_text, index),
            source_entity_type="membership",
            title=self._title(full_name, tier, patron_status),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["patreon", "membership", tier, patron_status, last_charge_status, address_country] if tag)),
            created_at=joined_at or last_charge_at or now,
            updated_at=updated_at,
        )

    def _title(self, full_name: str, tier: str, patron_status: str) -> str:
        title = full_name or "Patreon member"
        details = " - ".join(part for part in [tier, patron_status] if part)
        return f"{title} ({details})" if details else title

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = []
        for key, label in (
            ("full_name", "Member"),
            ("email", "Email"),
            ("tier", "Tier"),
            ("patron_status", "Status"),
            ("pledge_amount", "Pledge amount"),
            ("lifetime_amount", "Lifetime amount"),
            ("last_charge_date", "Last charge"),
            ("last_charge_status", "Last charge status"),
            ("next_charge_date", "Next charge"),
            ("address_country", "Country"),
            ("note", "Note"),
        ):
            if metadata.get(key) is not None:
                value = metadata[key]
                if key in {"pledge_amount", "lifetime_amount"} and metadata.get("currency"):
                    value = f"{value} {metadata['currency']}"
                parts.append(f"{label}: {value}")
        return "\n".join(parts)
