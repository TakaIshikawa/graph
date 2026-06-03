"""Adapter for GitHub Sponsors CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_money, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GitHubSponsorsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_sponsors_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["sponsor"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "sponsor" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        login = first(row, "Sponsor", "Sponsor Login", "login", "username")
        name = first(row, "Name", "Sponsor Name")
        email = first(row, "Email", "Sponsor Email")
        sponsorship_id = first(row, "Sponsorship ID", "id")
        if not any((login, name, email, sponsorship_id)):
            return None
        tier = first(row, "Tier", "Sponsorship Tier")
        amount = parse_money(first(row, "Amount", "Monthly Amount", "Amount in Cents"))
        currency = first(row, "Currency") or "USD"
        status = first(row, "Status") or ("ended" if first(row, "Ended At", "End Date") else "active")
        started_at = parse_datetime(first(row, "Started At", "Start Date", "Created At")) or datetime.now(timezone.utc)
        ended_at = parse_datetime(first(row, "Ended At", "End Date"))
        updated_at = ended_at or parse_datetime(first(row, "Updated At")) or started_at
        private = self._bool(first(row, "Private", "Is Private", "Privacy"))
        metadata = clean_metadata({"sponsorship_id": sponsorship_id, "login": login, "name": name, "email": email, "tier": tier, "amount": amount, "currency": currency, "status": status, "started_at": started_at.isoformat(), "ended_at": ended_at.isoformat() if ended_at else None, "private": private, "source_file": source_file})
        return KnowledgeUnit(source_project=self.name, source_id=digest_source_id(self.name, sponsorship_id or login or email or name, tier, started_at.date()), source_entity_type="sponsor", title=login or name or email or "GitHub sponsor", content=self._content(login, name, tier, amount, currency, status), content_type=ContentType.METADATA, metadata=metadata, tags=["github", "sponsor", status.casefold()], created_at=started_at, updated_at=updated_at)

    def _bool(self, value: str) -> bool | None:
        text = value.strip().casefold()
        if text in {"true", "1", "yes", "private"}:
            return True
        if text in {"false", "0", "no", "public"}:
            return False
        return None

    def _content(self, login: str, name: str, tier: str, amount: float | None, currency: str, status: str) -> str:
        parts = [login or name, f"Tier: {tier}" if tier else "", f"Amount: {amount} {currency}" if amount is not None else "", f"Status: {status}" if status else ""]
        return "\n".join(part for part in parts if part)
