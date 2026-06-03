"""Adapter for Todoist task JSON exports and API payloads."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class TodoistTasksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "todoist_tasks_json"

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
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("tasks", "items", "data", "results"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        task_id = _text(record.get("id") or record.get("task_id"))
        content = _text(record.get("content") or record.get("title") or record.get("name"))
        description = _text(record.get("description") or record.get("notes"))
        if not any((task_id, content, description)):
            return None
        created_at = parse_datetime(record.get("created_at") or record.get("createdAt"))
        completed_at = parse_datetime(record.get("completed_at") or record.get("completedAt"))
        updated_at = parse_datetime(record.get("updated_at") or record.get("updatedAt")) or completed_at or created_at
        labels = _labels(record.get("labels"))
        due = record.get("due")
        due_metadata = _due_metadata(due)
        project_name = _name(record.get("project")) or _text(record.get("project_name"))
        section_name = _name(record.get("section")) or _text(record.get("section_name"))
        completed = completed_at is not None or _bool(record.get("is_completed") or record.get("completed"))
        metadata = clean_metadata(
            {
                "task_id": task_id,
                "content": content,
                "description": description,
                "project_id": _id(record.get("project")) or _text(record.get("project_id")),
                "project_name": project_name,
                "section_id": _id(record.get("section")) or _text(record.get("section_id")),
                "section_name": section_name,
                "labels": labels,
                "priority": _int(record.get("priority")),
                "due": due_metadata,
                "due_date": due_metadata.get("date"),
                "completed": completed,
                "status": "completed" if completed else "active",
                "completed_at": completed_at.isoformat() if completed_at else _text(record.get("completed_at") or record.get("completedAt")),
                "created_at": created_at.isoformat() if created_at else _text(record.get("created_at") or record.get("createdAt")),
                "updated_at": updated_at.isoformat() if updated_at else _text(record.get("updated_at") or record.get("updatedAt")),
                "url": _text(record.get("url")),
                "parent_id": _text(record.get("parent_id") or record.get("parentId")),
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        title = content or f"Todoist task {task_id}"
        return KnowledgeUnit(
            source_project=SourceProject.TODOIST_TASKS_JSON,
            source_id=f"todoist_tasks_json:{task_id}" if task_id else digest_source_id("todoist_tasks_json", title, description, source_file, index),
            source_entity_type="task",
            title=title,
            content=_content(title, description, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=labels,
            created_at=created_at or completed_at or now,
            updated_at=updated_at or completed_at or created_at or now,
        )


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _name(value: Any) -> str:
    if isinstance(value, dict):
        return _text(value.get("name") or value.get("id"))
    return _text(value)


def _id(value: Any) -> str:
    if isinstance(value, dict):
        return _text(value.get("id"))
    return ""


def _labels(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    labels = [_name(item) for item in value]
    return [label for label in dict.fromkeys(labels) if label]


def _due_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return clean_metadata(
            {
                "date": _text(value.get("date")),
                "datetime": _text(value.get("datetime")),
                "string": _text(value.get("string")),
                "timezone": _text(value.get("timezone")),
                "is_recurring": _bool(value.get("is_recurring")),
            }
        )
    text = _text(value)
    return {"date": text} if text else {}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).casefold() in {"1", "true", "yes", "y", "done", "completed", "complete"}


def _int(value: Any) -> int | None:
    text = _text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _content(title: str, description: str, metadata: dict[str, Any]) -> str:
    parts = [title, description]
    for key, label in (("project_name", "Project"), ("section_name", "Section"), ("due_date", "Due"), ("url", "URL")):
        if metadata.get(key):
            parts.append(f"{label}: {metadata[key]}")
    return "\n".join(part for part in parts if part)
