"""Adapter for Google Contacts CSV exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GoogleContactsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_contacts_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["contact"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "contact" not in entity_types:
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: unit.source_id))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.csv") if child.is_file())

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        if not any(str(value or "").strip() for value in row.values()):
            return None

        name = self._first(row, "Name", "Full Name", "Given Name", "Family Name")
        emails = self._repeated(row, "E-mail", "Email")
        phones = self._repeated(row, "Phone")
        addresses = self._repeated(row, "Address")
        websites = self._repeated(row, "Website")
        groups = self._split_groups(self._first(row, "Group Membership", "Groups"))
        organization = self._organization(row)
        birthday = self._first(row, "Birthday", "Birthdate")
        notes = self._first(row, "Notes", "Note")
        updated = self._parse_datetime(self._first(row, "Updated", "Last Modified", "Modified"))
        created_at = updated or datetime.now(timezone.utc)

        if not name:
            name = emails[0] if emails else phones[0] if phones else f"Contact {index + 1}"

        metadata = {
            "name": name,
            "given_name": self._first(row, "Given Name"),
            "family_name": self._first(row, "Family Name"),
            "notes": notes,
            "emails": emails,
            "phones": phones,
            "addresses": addresses,
            "websites": websites,
            "organization": organization,
            "birthday": birthday,
            "groups": groups,
            "source_file": source_file,
        }
        if updated:
            metadata["updated"] = updated.isoformat()

        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_CONTACTS_CSV,
            source_id=self._source_id(row, name, emails, phones, source_file, index),
            source_entity_type="contact",
            title=name,
            content=self._content(name, notes, emails, phones, organization, addresses, birthday, groups),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["contact", *groups],
            created_at=created_at,
            updated_at=created_at,
        )

    def _repeated(self, row: dict[str, Any], *prefixes: str) -> list[str]:
        values: list[str] = []
        for key, value in row.items():
            key_text = str(key)
            if not any(key_text.lower().startswith(prefix.lower()) for prefix in prefixes):
                continue
            if (
                "value" not in key_text.lower()
                and "formatted" not in key_text.lower()
                and "email" not in key_text.lower()
                and key_text not in prefixes
            ):
                continue
            text = str(value or "").strip()
            if text and text not in values:
                values.append(text)
        return values

    def _organization(self, row: dict[str, Any]) -> dict[str, str]:
        return {
            "name": self._first(row, "Organization 1 - Name", "Organization Name", "Company"),
            "title": self._first(row, "Organization 1 - Title", "Job Title", "Title"),
            "department": self._first(row, "Organization 1 - Department", "Department"),
        }

    def _content(
        self,
        name: str,
        notes: str,
        emails: list[str],
        phones: list[str],
        organization: dict[str, str],
        addresses: list[str],
        birthday: str,
        groups: list[str],
    ) -> str:
        parts = [name]
        if notes:
            parts.append(notes)
        if emails:
            parts.append(f"Emails: {', '.join(emails)}")
        if phones:
            parts.append(f"Phones: {', '.join(phones)}")
        if organization.get("name"):
            parts.append(f"Organization: {organization['name']}")
        if addresses:
            parts.append(f"Addresses: {'; '.join(addresses)}")
        if birthday:
            parts.append(f"Birthday: {birthday}")
        if groups:
            parts.append(f"Groups: {', '.join(groups)}")
        return "\n".join(parts)

    def _source_id(
        self,
        row: dict[str, Any],
        name: str,
        emails: list[str],
        phones: list[str],
        source_file: str,
        index: int,
    ) -> str:
        explicit = self._first(row, "ID", "Contact ID")
        raw = explicit or "|".join([name.lower(), ",".join(emails), ",".join(phones), source_file, str(index)])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"google_contacts_csv:{digest}"

    def _split_groups(self, value: str) -> list[str]:
        if not value:
            return []
        cleaned = value.replace(":::*", "").replace("* myContacts", "My Contacts")
        return [item.strip() for item in cleaned.replace(";", ",").split(",") if item.strip()]

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).lower(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.lower())
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")))
        except ValueError:
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
