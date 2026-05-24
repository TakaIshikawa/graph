"""Adapter for Microsoft To Do tasks CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class MicrosoftTodoTasksCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "microsoft_todo_tasks_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["task"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "task" not in entity_types:
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Subject", "Task")
        notes = first(row, "Notes", "Note", "Body")
        if not title and not notes:
            return None
        list_name = first(row, "List Name", "List", "Folder")
        importance = first(row, "Importance", "Priority")
        due = parse_datetime(first(row, "Due Date", "Due", "Reminder Due Date"))
        completed = parse_datetime(first(row, "Completed Date", "Date Completed", "Completed On"))
        created = parse_datetime(first(row, "Created Date", "Created", "Creation Date"))
        updated = parse_datetime(first(row, "Modified Date", "Updated", "Last Modified"))
        status_text = first(row, "Status", "Completed", "Is Completed")
        status = self._status(status_text, completed)
        recurrence = clean_metadata(
            {
                "pattern": first(row, "Recurrence", "Repeat", "Repeat Pattern"),
                "interval": first(row, "Recurrence Interval", "Repeat Interval"),
                "until": first(row, "Recurrence End", "Repeat Until"),
            }
        )
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "title": title,
                "list_name": list_name,
                "status": status,
                "importance": importance,
                "due_date": due.isoformat() if due else "",
                "completed_date": completed.isoformat() if completed else "",
                "created_date": created.isoformat() if created else "",
                "notes": notes,
                "recurrence": recurrence,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        tags = [tag for tag in ("task", status, list_name, importance.casefold() if importance else "") if tag]
        return KnowledgeUnit(
            source_project="microsoft_todo_tasks_csv",
            source_id=digest_source_id("microsoft_todo_tasks_csv", first(row, "ID", "Task ID") or title, list_name, created.isoformat() if created else index),
            source_entity_type="task",
            title=title or "Untitled Microsoft To Do task",
            content=self._content(title, notes, list_name, status, importance, due, completed, recurrence),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=tags,
            created_at=created or due or completed or now,
            updated_at=updated or completed or due or created or now,
        )

    def _status(self, value: str, completed: datetime | None) -> str:
        text = value.casefold().strip()
        if completed or text in {"true", "yes", "1", "completed", "done"}:
            return "completed"
        return "incomplete"

    def _content(self, title: str, notes: str, list_name: str, status: str, importance: str, due: datetime | None, completed: datetime | None, recurrence: dict[str, Any]) -> str:
        parts = [title] if title else []
        parts.append(f"Status: {status}")
        if list_name:
            parts.append(f"List: {list_name}")
        if importance:
            parts.append(f"Importance: {importance}")
        if due:
            parts.append(f"Due: {due.isoformat()}")
        if completed:
            parts.append(f"Completed: {completed.isoformat()}")
        if recurrence:
            parts.append(f"Recurrence: {recurrence}")
        if notes:
            parts.append(notes)
        return "\n".join(parts)
