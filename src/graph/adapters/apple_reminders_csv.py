"""Adapter for Apple Reminders CSV exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class AppleRemindersCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "apple_reminders_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["reminder"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "reminder" not in entity_types:
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
                if sync_at and self._filter_datetime(unit) <= sync_at:
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
        return sorted(child for child in root.rglob("*.csv") if child.is_file())

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = self._first(row, "Title", "Name", "Reminder", "Task")
        notes = self._first(row, "Notes", "Note", "Body")
        list_name = self._first(row, "List", "List Name", "Calendar", "Group")
        if not title and not notes:
            return None
        title = title or "Untitled reminder"

        created = self._parse_datetime(self._first(row, "Created Date", "Creation Date", "Created", "Date Created"))
        due = self._parse_datetime(self._first(row, "Due Date", "Due", "Date Due", "Reminder Date"))
        completed = self._parse_datetime(
            self._first(row, "Completion Date", "Completed Date", "Date Completed", "Completed")
        )
        completed_flag = self._parse_bool(self._first(row, "Completed", "Is Completed", "Done"))
        status = "completed" if completed or completed_flag else "open"
        now = datetime.now(timezone.utc)
        created_at = created or due or completed or now

        priority = self._first(row, "Priority")
        url = self._first(row, "URL", "Url", "Link")
        metadata = {
            "title": title,
            "notes": notes,
            "list_name": list_name,
            "status": status,
            "completed": status == "completed",
            "priority": priority,
            "due_date": due.isoformat() if due else None,
            "completion_date": completed.isoformat() if completed else None,
            "created_date": created.isoformat() if created else None,
            "url": url,
            "source_file": source_file,
        }
        tags = ["reminder", status]
        if list_name:
            tags.append(list_name)

        return KnowledgeUnit(
            source_project=SourceProject.APPLE_REMINDERS_CSV,
            source_id=self._source_id(row, title, list_name, created_at, due, completed),
            source_entity_type="reminder",
            title=title,
            content=self._content(title, notes, due, url),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=tags,
            created_at=created_at,
            updated_at=completed or due or created_at,
        )

    def _filter_datetime(self, unit: KnowledgeUnit) -> datetime:
        for key in ("created_date", "due_date", "completion_date"):
            parsed = self._parse_datetime(unit.metadata.get(key))
            if parsed:
                return parsed
        return unit.created_at

    def _source_id(
        self,
        row: dict[str, Any],
        title: str,
        list_name: str,
        created: datetime,
        due: datetime | None,
        completed: datetime | None,
    ) -> str:
        explicit = self._first(row, "ID", "UUID", "Identifier")
        raw = explicit or "|".join(
            [title, list_name, created.isoformat(), due.isoformat() if due else "", completed.isoformat() if completed else ""]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"apple_reminders_csv:{digest}"

    def _content(self, title: str, notes: str, due: datetime | None, url: str) -> str:
        parts = [title]
        if notes:
            parts.append(notes)
        if due:
            parts.append(f"Due: {due.isoformat()}")
        if url:
            parts.append(f"URL: {url}")
        return "\n\n".join(parts)

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).lower(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.lower())
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_bool(self, value: Any) -> bool | None:
        if value is None or value == "":
            return None
        text = str(value).strip().lower()
        if text in {"true", "t", "yes", "y", "1", "done", "completed"}:
            return True
        if text in {"false", "f", "no", "n", "0", "open"}:
            return False
        return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        text = str(value).strip()
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        for fmt in (
            "%m/%d/%Y",
            "%m/%d/%y",
            "%m/%d/%Y %I:%M %p",
            "%m/%d/%y %I:%M %p",
            "%m/%d/%Y %H:%M",
            "%Y/%m/%d",
            "%b %d, %Y",
            "%B %d, %Y",
            "%b %d, %Y at %I:%M %p",
            "%B %d, %Y at %I:%M %p",
        ):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
