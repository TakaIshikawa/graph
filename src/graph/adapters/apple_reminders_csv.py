"""Adapter for Apple Reminders CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class AppleRemindersCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "apple_reminders_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["list", "reminder"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
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

        lists = self._list_units(units) if "list" in allowed_types else []
        if "list" in allowed_types:
            result.units.extend(lists)
        if "reminder" in allowed_types:
            result.units.extend(units)
        if "list" in allowed_types and "reminder" in allowed_types:
            result.edges.extend(self._contains_edges(lists, units))
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
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
        list_name = self._first(row, "List", "list", "List Name", "Reminder List", "Calendar", "Group")
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

        priority = self._first(row, "Priority", "priority")
        recurrence = self._recurrence_metadata(row)
        url = self._first(row, "URL", "Url", "Link")
        metadata = {
            "title": title,
            "notes": notes,
            "list_name": list_name,
            "status": status,
            "completed": status == "completed",
            "priority": priority,
            "recurrence": recurrence,
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
            content=self._content(title, notes, due, url, recurrence, priority),
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

    def _list_units(self, reminders: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for reminder in reminders:
            list_name = str(reminder.metadata.get("list_name") or "").strip()
            if list_name:
                key = self._normalize_list_name(list_name)
                names.setdefault(key, list_name)
                grouped.setdefault(key, []).append(reminder)

        now = datetime.now(timezone.utc)
        today = now.date()
        list_units: list[KnowledgeUnit] = []
        for key, list_reminders in grouped.items():
            list_name = names[key]
            open_count = sum(1 for reminder in list_reminders if reminder.metadata.get("status") == "open")
            completed_count = sum(1 for reminder in list_reminders if reminder.metadata.get("status") == "completed")
            due_dates = [
                due
                for reminder in list_reminders
                if (due := self._parse_datetime(reminder.metadata.get("due_date"))) is not None
            ]
            updated_dates = [reminder.updated_at for reminder in list_reminders]
            overdue_count = 0
            for reminder in list_reminders:
                due = self._parse_datetime(reminder.metadata.get("due_date"))
                if reminder.metadata.get("status") == "open" and due and due.date() < today:
                    overdue_count += 1
            list_units.append(
                KnowledgeUnit(
                    source_project=SourceProject.APPLE_REMINDERS_CSV,
                    source_id=self._list_source_id(list_name),
                    source_entity_type="list",
                    title=list_name,
                    content=f"Reminder list: {list_name}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "list_name": list_name,
                        "reminder_count": len(list_reminders),
                        "open_count": open_count,
                        "completed_count": completed_count,
                        "overdue_count": overdue_count,
                        "first_due_date": min(due_dates).isoformat() if due_dates else None,
                        "earliest_due_date": min(due_dates).isoformat() if due_dates else None,
                        "latest_due_date": max(due_dates).isoformat() if due_dates else None,
                        "last_updated_date": max(updated_dates).isoformat() if updated_dates else None,
                        "source_files": sorted({str(reminder.metadata.get("source_file")) for reminder in list_reminders}),
                    },
                    tags=["reminder-list", list_name],
                    created_at=min((reminder.created_at for reminder in list_reminders), default=now),
                    updated_at=max((reminder.updated_at for reminder in list_reminders), default=now),
                )
            )
        return list_units

    def _contains_edges(self, list_units: list[KnowledgeUnit], reminders: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        list_ids = {self._normalize_list_name(str(unit.metadata.get("list_name") or "")): unit.source_id for unit in list_units}
        edges: list[KnowledgeEdge] = []
        for reminder in reminders:
            list_id = list_ids.get(self._normalize_list_name(str(reminder.metadata.get("list_name") or "")))
            if not list_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=self._edge_id(list_id, reminder.source_id),
                    from_unit_id=list_id,
                    to_unit_id=reminder.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.APPLE_REMINDERS_CSV.value,
                        "relation_type": "list_contains_reminder",
                    },
                )
            )
        return list({edge.id: edge for edge in edges}.values())

    def _list_source_id(self, list_name: str) -> str:
        digest = hashlib.sha256(self._normalize_list_name(list_name).encode("utf-8")).hexdigest()[:24]
        return f"apple_reminders_csv:list:{digest}"

    def _normalize_list_name(self, list_name: str) -> str:
        return " ".join(list_name.strip().casefold().split())

    def _edge_id(self, list_source_id: str, reminder_source_id: str) -> str:
        digest = hashlib.sha256("|".join((list_source_id, reminder_source_id, "contains")).encode("utf-8")).hexdigest()[:24]
        return f"apple-reminders-csv-contains-{digest}"

    def _content(
        self,
        title: str,
        notes: str,
        due: datetime | None,
        url: str,
        recurrence: dict[str, str] | None = None,
        priority: str = "",
    ) -> str:
        parts = [title]
        if notes:
            parts.append(notes)
        if due:
            parts.append(f"Due: {due.isoformat()}")
        if priority:
            parts.append(f"Priority: {priority}")
        if recurrence:
            details = []
            if recurrence.get("repeat"):
                details.append(f"repeat {recurrence['repeat']}")
            if recurrence.get("repeat_interval"):
                details.append(f"every {recurrence['repeat_interval']}")
            if recurrence.get("repeat_end"):
                details.append(f"until {recurrence['repeat_end']}")
            if details:
                parts.append(f"Recurrence: {', '.join(details)}")
        if url:
            parts.append(f"URL: {url}")
        return "\n\n".join(parts)

    def _recurrence_metadata(self, row: dict[str, Any]) -> dict[str, str]:
        repeat = self._first(row, "Repeat", "Repeats", "Recurrence", "Recurrence Rule")
        interval = self._first(row, "Repeat Interval", "Interval", "Recurrence Interval")
        repeat_end = self._first(row, "Repeat End", "Repeat End Date", "Recurrence End")
        frequency = self._first(row, "Repeat Frequency", "Frequency")
        metadata = {
            "repeat": repeat,
            "repeat_interval": interval,
            "repeat_end": repeat_end,
            "frequency": frequency,
        }
        return {key: value for key, value in metadata.items() if value}

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).lower(): value for key, value in row.items()}
        compact = {self._normalize_field_name(str(key)): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.lower())
            if value is None:
                value = compact.get(self._normalize_field_name(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _normalize_field_name(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

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
