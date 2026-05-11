"""Adapter for generic Airtable CSV exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class AirtableCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "airtable_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["record"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "record" not in entity_types:
            return result
        sync_at = self._sync_datetime(since) if since else None

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
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit_from_row(self, row: dict[str, str], source_file: str, index: int) -> KnowledgeUnit | None:
        fields = {key: value for key, value in row.items() if value != ""}
        if not fields:
            return None
        title_field, title = self._title(row)
        content = self._content(row, title_field)
        created_text = self._first(row, "created_time", "Created time", "Created")
        modified_text = self._first(row, "last_modified_time", "Last modified time", "Last modified")
        created_at = self._parse_datetime(created_text)
        updated_at = self._parse_datetime(modified_text) or created_at
        now = datetime.now(timezone.utc)
        metadata = {
            "fields": fields,
            "title_field": title_field,
            "created_time": created_text,
            "last_modified_time": modified_text,
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project=SourceProject.AIRTABLE_CSV,
            source_id=f"airtable_csv:{self._digest(source_file, index, fields)}",
            source_entity_type="record",
            title=title,
            content=content,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["airtable"],
            created_at=created_at or updated_at or now,
            updated_at=updated_at or created_at or now,
        )

    def _title(self, row: dict[str, str]) -> tuple[str, str]:
        for preferred in ("Name", "Title", "Summary"):
            value = row.get(preferred)
            if value and value.strip():
                return preferred, value.strip()
        for key, value in row.items():
            if value and value.strip():
                return key, value.strip()
        return "", "Untitled Airtable record"

    def _content(self, row: dict[str, str], title_field: str) -> str:
        parts = []
        for key, value in row.items():
            text = value.strip() if value else ""
            if text and key != title_field:
                parts.append(f"{key}: {text}")
        return "\n".join(parts) or row.get(title_field, "").strip() or "Untitled Airtable record"

    def _iter_paths(self) -> list[Path]:
        path = Path(self.path).expanduser() if self.path else None
        if path is None:
            return []
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(child for child in path.rglob("*.csv") if child.is_file())
        return []

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return [
                {str(key).strip(): (value or "").strip() for key, value in row.items() if key is not None}
                for row in reader
            ]

    def _first(self, row: dict[str, str], *keys: str) -> str:
        lower = {key.lower(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key) or lower.get(key.lower())
            if value and value.strip():
                return value.strip()
        return ""

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _digest(self, source_file: str, index: int, fields: dict[str, str]) -> str:
        payload = repr((source_file, index, sorted(fields.items())))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
