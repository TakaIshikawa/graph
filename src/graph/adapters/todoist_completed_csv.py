"""Adapter for Todoist completed task CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class TodoistCompletedCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "todoist_completed_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["completed_task"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "completed_task" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows, start=2):
                unit = self._unit(row, path.name, index)
                if unit is None or (sync_at and unit.updated_at <= sync_at):
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, source_row: int) -> KnowledgeUnit | None:
        task_id = first(row, "Task ID", "ID", "task_id")
        content = first(row, "Content", "Task", "Task Content", "Name")
        project = first(row, "Project", "Project Name")
        completed = parse_datetime(first(row, "Completed", "Completed At", "Completed Date"))
        if not any([task_id, content, project, completed]):
            return None
        created = parse_datetime(first(row, "Created", "Created At", "Created Date"))
        due = parse_datetime(first(row, "Due", "Due Date"))
        url = first(row, "URL", "Link")
        labels = split_values(first(row, "Labels", "Label"))
        metadata = clean_metadata({"task_id": task_id, "content": content, "project": project, "section": first(row, "Section"), "labels": labels, "priority": parse_int(first(row, "Priority")), "completed_at": completed.isoformat() if completed else first(row, "Completed", "Completed At"), "created_at": created.isoformat() if created else first(row, "Created", "Created At"), "due_date": due.date().isoformat() if due else first(row, "Due", "Due Date"), "url": url, "source_url": url, "external_url": url, "recurring": self._bool(first(row, "Recurring", "Is Recurring")), "source_file": source_file, "source_row": source_row})
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{task_id}" if task_id else digest_source_id(self.name, content, project, completed), source_entity_type="completed_task", title=content or "Todoist completed task", content=self._content(content, metadata), content_type=ContentType.METADATA, metadata=metadata, tags=list(dict.fromkeys(["todoist", "completed", project, *labels])), created_at=created or completed or now, updated_at=completed or created or now)

    def _bool(self, value: Any) -> bool | None:
        text = str(value or "").strip().casefold()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
        return None

    def _content(self, content: str, metadata: dict[str, Any]) -> str:
        parts = [content or "Todoist completed task"]
        for key, label in (("project", "Project"), ("section", "Section"), ("completed_at", "Completed"), ("due_date", "Due"), ("url", "URL")):
            if key in metadata:
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
