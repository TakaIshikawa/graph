"""Adapter for Todoist task CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class TodoistTasksCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "todoist_tasks_csv"

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
        if sync_at and sync_at.tzinfo is None:
            sync_at = sync_at.replace(tzinfo=timezone.utc)
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        content = first(row, "content", "task", "task name", "name")
        description = first(row, "description", "comments", "comment")
        if not any((content, description)):
            return None
        completed = _truthy(first(row, "completed", "is completed", "done")) or bool(first(row, "completed at", "date completed"))
        created_at = parse_datetime(first(row, "created at", "date added", "created")) or datetime.now(timezone.utc)
        completed_at = parse_datetime(first(row, "completed at", "date completed"))
        updated_at = completed_at or created_at
        labels = [label.casefold() for label in split_values(first(row, "labels", "label"))]
        metadata = clean_metadata(
            {
                "task_id": first(row, "id", "task id"),
                "project": first(row, "project", "project name"),
                "section": first(row, "section", "section name"),
                "labels": labels,
                "priority": first(row, "priority"),
                "due_date": first(row, "due date", "due"),
                "completed": completed,
                "status": "completed" if completed else "active",
                "created_at": created_at.isoformat(),
                "completed_at": completed_at.isoformat() if completed_at else None,
                "description": description,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.TODOIST_TASKS_CSV,
            source_id=digest_source_id("todoist_tasks_csv", metadata.get("task_id") or content, metadata.get("project"), index),
            source_entity_type="task",
            title=content or "Todoist task",
            content="\n".join(part for part in (content, description) if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=labels,
            created_at=created_at,
            updated_at=updated_at,
        )


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y", "done", "completed", "complete"}
