"""Adapter for Things 3 CSV task exports."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class ThingsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "things_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["task"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "task" not in entity_types:
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.created_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.created_at, unit.source_id)))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if not root.is_dir():
            return []
        return sorted(root.glob("*.csv"), key=lambda child: child.name)

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def _unit_from_row(self, row: dict[str, str], source_file: str) -> KnowledgeUnit | None:
        title = self._first(row, "Title", "title", "Task", "Name")
        if not title:
            return None
        notes = self._first(row, "Notes", "notes", "Note")
        created_at = self._parse_datetime(self._first(row, "Creation Date", "Created", "created_at"))
        completed_at = self._parse_datetime(self._first(row, "Completion Date", "Completed", "completed_at"))
        if created_at is None:
            created_at = completed_at or datetime.now(timezone.utc)

        canceled = self._parse_bool(self._first(row, "Canceled", "Cancelled", "canceled"))
        status = self._status(row, completed_at, canceled)
        tags = self._split_tags(self._first(row, "Tags", "tags"))
        metadata = {
            "title": title,
            "notes": notes,
            "area": self._first(row, "Area", "area"),
            "project": self._first(row, "Project", "project"),
            "tags": tags,
            "status": status,
            "creation_date": created_at.isoformat() if created_at else None,
            "start_date": self._datetime_metadata(self._first(row, "Start Date", "When", "start_date")),
            "deadline": self._datetime_metadata(self._first(row, "Deadline", "Due Date", "deadline")),
            "completion_date": completed_at.isoformat() if completed_at else None,
            "canceled": canceled,
            "checklist": self._checklist(self._first(row, "Checklist", "checklist")),
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project=SourceProject.THINGS_CSV,
            source_id=self._source_id(row, title, created_at),
            source_entity_type="task",
            title=title,
            content=self._content(title, notes),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["things", "task", *tags],
            created_at=created_at,
            updated_at=completed_at or created_at,
        )

    def _status(self, row: dict[str, str], completed_at: datetime | None, canceled: bool | None) -> str:
        explicit = self._first(row, "Status", "status")
        if explicit:
            return explicit.lower()
        if canceled:
            return "canceled"
        if completed_at:
            return "completed"
        if self._first(row, "Start Date", "When", "start_date"):
            return "scheduled"
        return "open"

    def _source_id(self, row: dict[str, str], title: str, created_at: datetime) -> str:
        explicit = self._first(row, "UUID", "ID", "id", "uuid")
        identifier = explicit or "|".join([title, created_at.isoformat(), self._first(row, "Project", "project")])
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:24]
        return f"things_csv:{digest}"

    def _content(self, title: str, notes: str) -> str:
        return f"{title}\n\n{notes}".strip()

    def _checklist(self, value: str) -> Any:
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else value
        except json.JSONDecodeError:
            return [item.strip() for item in value.splitlines() if item.strip()]

    def _split_tags(self, value: str) -> list[str]:
        if not value:
            return []
        normalized = value.replace(";", ",")
        return [tag.strip() for tag in normalized.split(",") if tag.strip()]

    def _first(self, row: dict[str, str], *keys: str) -> str:
        lowered = {key.lower(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.lower())
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _datetime_metadata(self, value: str) -> str | None:
        parsed = self._parse_datetime(value)
        return parsed.isoformat() if parsed else None

    def _parse_bool(self, value: str) -> bool | None:
        if not value:
            return None
        text = value.strip().lower()
        if text in {"true", "t", "yes", "y", "1", "canceled", "cancelled"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
        return None

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        for candidate in (value, f"{value}T00:00:00"):
            try:
                parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
                return self._ensure_utc(parsed)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
