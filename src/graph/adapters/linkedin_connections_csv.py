"""Adapter for LinkedIn connections CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LinkedInConnectionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "linkedin_connections_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["connection"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "connection" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = self._rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _rows(self, path: Path) -> list[dict[str, str]]:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        header_index = next((i for i, line in enumerate(lines) if "First Name" in line and "Last Name" in line), 0)
        reader = csv.DictReader(lines[header_index:])
        return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader if row]

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        first_name = first(row, "First Name", "first_name")
        last_name = first(row, "Last Name", "last_name")
        name = " ".join(part for part in (first_name, last_name) if part).strip() or first(row, "Name", "Full Name")
        company = first(row, "Company", "Organization")
        position = first(row, "Position", "Title")
        email = first(row, "Email Address", "Email")
        url = first(row, "Profile URL", "URL")
        notes = first(row, "Notes", "Note")
        if not any((name, company, email, url)):
            return None
        connected_at = parse_datetime(first(row, "Connected On", "Connected Date", "Date")) or datetime.now(timezone.utc)
        metadata = clean_metadata({"first_name": first_name, "last_name": last_name, "name": name, "company": company, "position": position, "email": email, "profile_url": url, "notes": notes, "connected_at": connected_at.isoformat(), "source_file": source_file})
        return KnowledgeUnit(source_project=self.name, source_id=digest_source_id(self.name, email or url or name, company, index if not (email or url or name) else ""), source_entity_type="connection", title=name or email or company, content=self._content(name, company, position, email, url, notes), content_type=ContentType.METADATA, metadata=metadata, tags=["linkedin", "connection", *([company] if company else [])], created_at=connected_at, updated_at=connected_at)

    def _content(self, name: str, company: str, position: str, email: str, url: str, notes: str) -> str:
        return "\n".join(part for part in (name, f"Position: {position}" if position else "", f"Company: {company}" if company else "", f"Email: {email}" if email else "", f"URL: {url}" if url else "", notes) if part)
