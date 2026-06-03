"""Adapter for Google Tasks Takeout JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GoogleTasksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_tasks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["task"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "task" not in set(entity_types or self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                task_lists = self._read_task_lists(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for list_title, task in task_lists:
                unit = self._unit_from_task(task, list_title, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _iter_paths(self) -> list[Path]:
        root = Path(self.path).expanduser()
        if not self.path:
            return []
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.json") if path.is_file())

    def _read_task_lists(self, path: Path) -> list[tuple[str, dict[str, Any]]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        pairs: list[tuple[str, dict[str, Any]]] = []
        for task_list in self._lists(parsed):
            title = self._text(task_list.get("title") or task_list.get("name") or task_list.get("kind"))
            for task in self._tasks(task_list):
                pairs.append((title, task))
        return pairs

    def _lists(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("task_lists", "taskLists", "lists"):
                if isinstance(value.get(key), list):
                    return [item for item in value[key] if isinstance(item, dict)]
            items = value.get("items")
            if isinstance(items, list) and any(isinstance(item, dict) and ("tasks" in item or "items" in item) for item in items):
                return [item for item in items if isinstance(item, dict)]
            return [value]
        return []

    def _tasks(self, task_list: dict[str, Any]) -> list[dict[str, Any]]:
        raw = task_list.get("tasks") or task_list.get("items") or []
        tasks = [item for item in raw if isinstance(item, dict)]
        flattened: list[dict[str, Any]] = []
        for task in tasks:
            flattened.extend(self._flatten(task, None))
        return flattened

    def _flatten(self, task: dict[str, Any], parent_id: str | None) -> list[dict[str, Any]]:
        task = dict(task)
        if parent_id and not task.get("parent"):
            task["parent"] = parent_id
        current_id = self._text(task.get("id"))
        children = task.pop("children", None) or task.pop("subtasks", None) or []
        flattened = [task]
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    flattened.extend(self._flatten(child, current_id or parent_id))
        return flattened

    def _unit_from_task(self, task: dict[str, Any], list_title: str, source_file: str) -> KnowledgeUnit | None:
        title = self._text(task.get("title"))
        notes = self._text(task.get("notes") or task.get("description"))
        task_id = self._text(task.get("id"))
        if not title and not notes:
            return None
        updated = parse_datetime(task.get("updated") or task.get("updated_at"))
        due = parse_datetime(task.get("due") or task.get("due_date"))
        completed = parse_datetime(task.get("completed") or task.get("completed_at"))
        links = self._links(task.get("links"))
        metadata = {
            "task_list": list_title,
            "task_id": task_id,
            "title": title,
            "notes": notes,
            "status": self._text(task.get("status")),
            "due_at": due.isoformat() if due else self._text(task.get("due") or task.get("due_date")),
            "completed_at": completed.isoformat() if completed else self._text(task.get("completed") or task.get("completed_at")),
            "updated_at": updated.isoformat() if updated else self._text(task.get("updated") or task.get("updated_at")),
            "parent_task_id": self._text(task.get("parent")),
            "links": links,
            "source_file": source_file,
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_TASKS_JSON,
            source_id=f"google_tasks_json:{task_id}" if task_id else digest_source_id("google_tasks_json", list_title, title, notes),
            source_entity_type="task",
            title=title or notes,
            content=self._content(title, notes, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["google_tasks", "task", list_title, metadata["status"]] if tag)),
            created_at=due or completed or updated or now,
            updated_at=updated or completed or due or now,
        )

    def _links(self, value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        links = []
        for item in value:
            if isinstance(item, dict):
                link = clean_metadata({"type": self._text(item.get("type")), "description": self._text(item.get("description")), "link": self._text(item.get("link") or item.get("url"))})
                if link:
                    links.append(link)
        return links

    def _content(self, title: str, notes: str, metadata: dict[str, Any]) -> str:
        parts = [title, notes]
        if metadata.get("task_list"):
            parts.append(f"List: {metadata['task_list']}")
        if metadata.get("due_at"):
            parts.append(f"Due: {metadata['due_at']}")
        return "\n".join(part for part in parts if part)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
