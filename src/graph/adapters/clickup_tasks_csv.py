"""Adapter for ClickUp tasks CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class ClickUpTasksCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "clickup_tasks_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["task"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "task" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        name = first(row, "Task Name", "Name", "Title")
        if not name:
            return None
        task_id = first(row, "Task ID", "ID")
        status = first(row, "Status")
        priority = first(row, "Priority")
        assignees = split_values(first(row, "Assignees", "Assignee"))
        tags = split_values(first(row, "Tags", "Tag"))
        created_text = first(row, "Date Created", "Created")
        updated_text = first(row, "Date Updated", "Updated")
        due_text = first(row, "Due Date", "Due")
        created_at = parse_datetime(created_text)
        updated_at = parse_datetime(updated_text)
        now = datetime.now(timezone.utc)
        url = first(row, "URL", "Url", "Link")
        description = first(row, "Description", "Notes")
        metadata = clean_metadata(
            {
                "task_id": task_id,
                "status": status,
                "priority": priority,
                "assignees": assignees,
                "tags": tags,
                "date_created": created_at.isoformat() if created_at else created_text,
                "date_updated": updated_at.isoformat() if updated_at else updated_text,
                "due_date": due_text,
                "url": url,
                "description": description,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="clickup_tasks_csv",
            source_id=digest_source_id("clickup_tasks_csv", task_id or name, "" if task_id else index),
            source_entity_type="task",
            title=name,
            content="\n".join(part for part in [name, description, f"Status: {status}" if status else "", f"URL: {url}" if url else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at or updated_at or now,
            updated_at=updated_at or created_at or now,
        )
