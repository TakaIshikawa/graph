"""Adapter for Asana task JSON exports and API payloads."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime
from graph.adapters.asana_tasks_csv import AsanaTasksCsvAdapter
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AsanaTasksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "asana_tasks_json"

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
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit(record, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("data", "tasks", "items"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        task_id = _first(record, "gid", "id", "task_id")
        name = _first(record, "name", "title")
        notes = _first(record, "notes", "description")
        if not any([task_id, name, notes]):
            return None
        created = parse_datetime(_first(record, "created_at", "createdAt", "created"))
        modified = parse_datetime(_first(record, "modified_at", "modifiedAt", "updated_at", "updatedAt")) or created
        completed_at = parse_datetime(_first(record, "completed_at", "completedAt"))
        due_on = _first(record, "due_on", "dueOn")
        start_on = _first(record, "start_on", "startOn")
        completed = _bool(record.get("completed"))
        status = AsanaTasksCsvAdapter()._normalize_status("completed" if completed or completed_at else _first(record, "status") or "open")
        projects = [_name(item) for item in record.get("projects", []) if _name(item)] if isinstance(record.get("projects"), list) else []
        tags = [_name(item) for item in record.get("tags", []) if _name(item)] if isinstance(record.get("tags"), list) else []
        workspace = _name(record.get("workspace"))
        assignee = _name(record.get("assignee"))
        permalink = _first(record, "permalink_url", "permalinkUrl", "url")
        metadata = clean_metadata(
            {
                "task_id": task_id,
                "gid": task_id,
                "name": name,
                "notes": notes,
                "permalink": permalink,
                "url": permalink,
                "status": status,
                "completed": completed if completed is not None else (True if completed_at else None),
                "due_on": due_on,
                "start_on": start_on,
                "project": projects[0] if projects else _name(record.get("project")),
                "projects": projects or ([_name(record.get("project"))] if _name(record.get("project")) else []),
                "workspace": workspace,
                "assignee": assignee,
                "tags": tags,
                "created_at": created.isoformat() if created else _first(record, "created_at", "createdAt"),
                "modified_at": modified.isoformat() if modified else _first(record, "modified_at", "modifiedAt", "updated_at", "updatedAt"),
                "completed_at": completed_at.isoformat() if completed_at else _first(record, "completed_at", "completedAt"),
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        graph_tags = list(dict.fromkeys(["asana", "task", *(metadata.get("projects") or []), *tags]))
        title = name or f"Asana task {task_id}"
        return KnowledgeUnit(
            source_project="asana_tasks_json",
            source_id=f"asana_tasks_json:{task_id}" if task_id else digest_source_id("asana_tasks_json", title, notes, created, source_file, index),
            source_entity_type="task",
            title=title,
            content=_content(title, notes, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=[tag for tag in graph_tags if tag],
            created_at=created or modified or now,
            updated_at=modified or completed_at or created or now,
        )


def _first(row: dict[str, Any], *keys: str) -> str:
    lowered = {str(key).casefold(): value for key, value in row.items()}
    for key in keys:
        value = row.get(key, lowered.get(key.casefold()))
        if value is not None and not isinstance(value, (dict, list)) and str(value).strip():
            return str(value).strip()
    return ""


def _name(value: Any) -> str:
    if isinstance(value, dict):
        return _first(value, "name", "gid", "id")
    return str(value).strip() if value is not None else ""


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = "" if value is None else str(value).strip().casefold()
    if text in {"true", "yes", "1", "completed"}:
        return True
    if text in {"false", "no", "0", "incomplete", "open"}:
        return False
    return None


def _content(title: str, notes: str, metadata: dict[str, Any]) -> str:
    parts = [title]
    if notes:
        parts.append(notes)
    for key, label in (("assignee", "Assignee"), ("status", "Status"), ("due_on", "Due"), ("permalink", "URL")):
        if metadata.get(key):
            parts.append(f"{label}: {metadata[key]}")
    return "\n".join(parts)
