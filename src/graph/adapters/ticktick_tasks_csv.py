"""Adapter for TickTick task CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class TickTickTasksCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "ticktick_tasks_csv"

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
                unit = self._unit_from_row(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        task_id = first(row, "ID", "Id", "Task ID", "Task Id")
        title = first(row, "Title", "Task", "Name")
        content = first(row, "Content", "Description", "Notes", "Note")
        list_name = first(row, "List Name", "List", "Project")
        tags = split_values(first(row, "Tags", "Tag"))
        priority = self._normalize_priority(first(row, "Priority"))
        status = self._normalize_status(first(row, "Status", "Completed"))
        timezone_name = first(row, "Timezone", "Time Zone")
        created_text = first(row, "Created Time", "Created At", "Created")
        modified_text = first(row, "Modified Time", "Updated Time", "Updated At", "Modified At")
        due_text = first(row, "Due Date", "Due Time", "Due At", "Due")
        completed_text = first(row, "Completed Time", "Completed At", "Completion Time")

        if not any([task_id, title, content, list_name, tags, priority, status, timezone_name, created_text, modified_text, due_text, completed_text]):
            return None

        created_at = parse_datetime(created_text)
        modified_at = parse_datetime(modified_text)
        due_at = parse_datetime(due_text)
        completed_at = parse_datetime(completed_text)
        if completed_at and not status:
            status = "completed"
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "task_id": task_id,
                "title": title,
                "content": content,
                "list_name": list_name,
                "tags": tags,
                "priority": priority,
                "status": status,
                "created_at": created_at.isoformat() if created_at else created_text,
                "modified_at": modified_at.isoformat() if modified_at else modified_text,
                "due_at": due_at.isoformat() if due_at else due_text,
                "completed_at": completed_at.isoformat() if completed_at else completed_text,
                "timezone": timezone_name,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        timestamp = created_at or due_at or completed_at or modified_at or now
        sync_timestamp = modified_at or completed_at or due_at or created_at or now
        return KnowledgeUnit(
            source_project="ticktick_tasks_csv",
            source_id=f"ticktick_tasks_csv:{task_id}" if task_id else digest_source_id("ticktick_tasks_csv", title, content, list_name, due_text, created_text, source_file, index),
            source_entity_type="task",
            title=title or "TickTick task",
            content=self._content(title, content, metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["ticktick", "task", list_name, status, priority, *tags] if tag)),
            created_at=timestamp,
            updated_at=sync_timestamp,
        )

    def _normalize_priority(self, value: str) -> str:
        text = " ".join(value.strip().casefold().split())
        aliases = {
            "0": "none",
            "1": "low",
            "2": "medium",
            "3": "high",
            "5": "medium",
            "none": "none",
            "low priority": "low",
            "medium priority": "medium",
            "high priority": "high",
        }
        return aliases.get(text, text)

    def _normalize_status(self, value: str) -> str:
        text = " ".join(value.strip().casefold().split())
        aliases = {
            "0": "open",
            "1": "completed",
            "2": "completed",
            "yes": "completed",
            "true": "completed",
            "done": "completed",
            "complete": "completed",
            "no": "open",
            "false": "open",
            "todo": "open",
            "incomplete": "open",
        }
        return aliases.get(text, text)

    def _content(self, title: str, content: str, metadata: dict[str, Any]) -> str:
        parts = [
            title,
            content,
            f"List: {metadata.get('list_name')}" if metadata.get("list_name") else "",
            f"Status: {metadata.get('status')}" if metadata.get("status") else "",
            f"Priority: {metadata.get('priority')}" if metadata.get("priority") else "",
            f"Due: {metadata.get('due_at')}" if metadata.get("due_at") else "",
        ]
        return "\n".join(part for part in parts if part)
