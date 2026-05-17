"""Adapter for Habitica task CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class HabiticaTasksCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "habitica_tasks_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["task"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "task" not in entity_types:
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
        task_id = first(row, "Task ID", "Task Id", "ID", "Id")
        task_type = first(row, "Type", "Task Type")
        text = first(row, "Text", "Title", "Task")
        notes = first(row, "Notes", "Note", "Description")
        tags = split_values(first(row, "Tags", "Tag Names"))
        priority = parse_float(first(row, "Priority"))
        difficulty = first(row, "Difficulty")
        value = parse_float(first(row, "Value"))
        due_text = first(row, "Due Date", "Due", "Due At")
        created_text = first(row, "Created At", "Created")
        updated_text = first(row, "Updated At", "Updated", "Modified At")
        completed_text = first(row, "Completed At", "Completed Date")
        status = self._status(first(row, "Status", "Completed"), completed_text)
        checklist = first(row, "Checklist", "Checklist Text", "Subtasks")

        if not any([task_id, task_type, text, notes, tags, priority is not None, difficulty, value is not None, due_text, created_text, updated_text, completed_text, checklist]):
            return None

        created_at = parse_datetime(created_text)
        updated_at = parse_datetime(updated_text)
        due_at = parse_datetime(due_text)
        completed_at = parse_datetime(completed_text)
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "task_id": task_id,
                "type": task_type,
                "text": text,
                "notes": notes,
                "tags": tags,
                "priority": priority,
                "difficulty": difficulty,
                "value": value,
                "due_date": due_at.isoformat() if due_at else due_text,
                "created_at": created_at.isoformat() if created_at else created_text,
                "updated_at": updated_at.isoformat() if updated_at else updated_text,
                "completed_at": completed_at.isoformat() if completed_at else completed_text,
                "status": status,
                "checklist": checklist,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        timestamp = created_at or updated_at or completed_at or due_at or now
        modified = updated_at or completed_at or created_at or due_at or now
        return KnowledgeUnit(
            source_project="habitica_tasks_csv",
            source_id=f"habitica_tasks_csv:{task_id}" if task_id else digest_source_id("habitica_tasks_csv", task_type, text, notes, due_text, created_text, index),
            source_entity_type="task",
            title=text or (f"Habitica {task_type} task" if task_type else "Habitica task"),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["habitica", task_type, status, *tags] if tag)),
            created_at=timestamp,
            updated_at=modified,
        )

    def _status(self, status: str, completed_at: str) -> str:
        text = status.strip().casefold()
        if completed_at.strip():
            return "completed"
        if text in {"completed", "complete", "done", "true", "yes", "1"}:
            return "completed"
        if text in {"open", "active", "incomplete", "todo", "false", "no", "0"}:
            return "open"
        return text

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [
            metadata.get("text", ""),
            metadata.get("notes", ""),
            f"Type: {metadata.get('type')}" if metadata.get("type") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Due: {metadata.get('due_date')}" if metadata.get("due_date") else "",
            f"Checklist: {metadata.get('checklist')}" if metadata.get("checklist") else "",
        ]
        return "\n".join(part for part in parts if part)
