"""Adapter for Google Tasks JSON exports from Google Takeout."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class GoogleTasksAdapter(SourceAdapter):
    """Import Google Tasks Takeout JSON exports."""

    @property
    def name(self) -> str:
        return "google_tasks"

    @property
    def entity_types(self) -> list[str]:
        return ["task", "task_list", "task_due_day"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if not self.path:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        allowed_types = set(entity_types) if entity_types else None
        files = self._json_paths(root)
        task_units: list[KnowledgeUnit] = []

        for file_path in files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue

            if not isinstance(data, dict):
                continue

            list_title = self._list_title(data, file_path)
            list_source_id = self._list_source_id(data, file_path, list_title)

            # Create task_list unit
            if not allowed_types or "task_list" in allowed_types:
                unit = KnowledgeUnit(
                    source_project=SourceProject.GOOGLE_TASKS,
                    source_id=list_source_id,
                    source_entity_type="task_list",
                    title=list_title,
                    content=list_title,
                    content_type=ContentType.ARTIFACT,
                    metadata={"list_title": list_title, "source_file": file_path.name},
                    tags=[],
                )
                result.units.append(unit)

            # Process tasks
            items = data.get("items") or []
            task_id_map: dict[str, str] = {}  # google task id -> source_id

            for item in items:
                if not isinstance(item, dict):
                    continue

                task_title = item.get("title") or ""
                task_notes = item.get("notes") or ""
                task_id_raw = item.get("id") or task_title
                if not task_id_raw:
                    continue

                digest = hashlib.sha1(str(task_id_raw).encode("utf-8")).hexdigest()[:16]
                source_id = f"google_tasks:task:{digest}"
                task_id_map[str(task_id_raw)] = source_id

                status = (item.get("status") or "").lower()
                tags: list[str] = []
                if status:
                    tags.append(status)

                metadata: dict[str, Any] = {
                    "list_title": list_title,
                    "list_source_id": list_source_id,
                }
                if status:
                    metadata["status"] = status
                if item.get("due"):
                    metadata["due"] = item["due"]
                if item.get("updated"):
                    metadata["updated"] = item["updated"]
                if item.get("completed"):
                    metadata["completed"] = item["completed"]
                if item.get("parent"):
                    metadata["parent_id"] = item["parent"]
                if item.get("links"):
                    metadata["links"] = item["links"]
                recurrence = self._recurrence_metadata(item)
                if recurrence not in (None, "", [], {}):
                    metadata["recurrence"] = recurrence

                created_at = self._parse_datetime(item.get("updated")) or datetime.now(timezone.utc)

                unit = KnowledgeUnit(
                    source_project=SourceProject.GOOGLE_TASKS,
                    source_id=source_id,
                    source_entity_type="task",
                    title=task_title or "Untitled Task",
                    content=task_notes or task_title or "",
                    content_type=ContentType.ARTIFACT,
                    metadata=metadata,
                    tags=sorted(tags),
                    created_at=created_at,
                    updated_at=created_at,
                )
                task_units.append(unit)
                if not allowed_types or "task" in allowed_types:
                    result.units.append(unit)
                if not allowed_types or {"task", "task_list"}.issubset(allowed_types):
                    result.edges.append(
                        KnowledgeEdge(
                            id=self._edge_id(source_id, list_source_id, "task_list_membership"),
                            from_unit_id=source_id,
                            to_unit_id=list_source_id,
                            relation=EdgeRelation.CONTAINS,
                            source=EdgeSource.SOURCE,
                            metadata={
                                "source_project": SourceProject.GOOGLE_TASKS.value,
                                "relation_type": "task_list_membership",
                                "list_title": list_title,
                            },
                        )
                    )

            # Create hierarchy edges for subtasks
            if not allowed_types or "task" in allowed_types:
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    parent_raw = item.get("parent")
                    if not parent_raw:
                        continue
                    child_id_raw = item.get("id") or ""
                    if not child_id_raw:
                        continue
                    parent_source = task_id_map.get(str(parent_raw))
                    child_source = task_id_map.get(str(child_id_raw))
                    if parent_source and child_source:
                        edge_id = self._edge_id(parent_source, child_source, "contains")
                        result.edges.append(
                            KnowledgeEdge(
                                id=edge_id,
                                from_unit_id=parent_source,
                                to_unit_id=child_source,
                                relation=EdgeRelation.CONTAINS,
                                source=EdgeSource.SOURCE,
                                metadata={
                                    "source_project": SourceProject.GOOGLE_TASKS.value,
                                    "relation_type": "subtask",
                                },
                            )
                        )

        if not allowed_types or "task_due_day" in allowed_types:
            due_day_units = self._due_day_units(task_units)
            result.units.extend(due_day_units)
            if not allowed_types or {"task_due_day", "task"}.issubset(allowed_types):
                result.edges.extend(self._due_day_edges(due_day_units, task_units))

        result.units.sort(key=lambda u: (u.source_entity_type, u.source_id))
        result.edges.sort(key=lambda e: (e.from_unit_id, e.to_unit_id))
        return result

    def _json_paths(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root]
        return sorted(p for p in root.rglob("*.json") if p.is_file())

    def _list_title(self, data: dict[str, Any], file_path: Path) -> str:
        title = str(data.get("title") or "").strip()
        return title or file_path.stem or "Untitled Task List"

    def _list_source_id(self, data: dict[str, Any], file_path: Path, list_title: str) -> str:
        raw_id = str(data.get("id") or "").strip()
        stable_key = raw_id or file_path.as_posix() or list_title
        digest = hashlib.sha1(f"list:{stable_key}".encode("utf-8")).hexdigest()[:16]
        return f"google_tasks:task_list:{digest}"

    def _edge_id(self, from_id: str, to_id: str, relation_type: str) -> str:
        raw = "|".join([SourceProject.GOOGLE_TASKS.value, relation_type, from_id, to_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"google_tasks-{relation_type}-{digest}"

    def _due_day_units(self, tasks: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for task in tasks:
            due_day = self._due_day(task.metadata.get("due"))
            if due_day:
                grouped.setdefault(due_day, []).append(task)

        units: list[KnowledgeUnit] = []
        for due_day, day_tasks in sorted(grouped.items()):
            ordered = sorted(day_tasks, key=lambda task: task.source_id)
            completed_count = sum(1 for task in ordered if str(task.metadata.get("status", "")).lower() == "completed")
            list_titles = sorted({str(task.metadata.get("list_title")) for task in ordered if task.metadata.get("list_title")})
            source_id = self._due_day_source_id(due_day)
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GOOGLE_TASKS,
                    source_id=source_id,
                    source_entity_type="task_due_day",
                    title=f"Google Tasks due {due_day}",
                    content=f"{len(ordered)} Google Tasks due on {due_day}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "due_date": due_day,
                        "task_count": len(ordered),
                        "completed_count": completed_count,
                        "incomplete_count": len(ordered) - completed_count,
                        "list_titles": list_titles,
                        "task_source_ids": [task.source_id for task in ordered],
                    },
                    tags=["google_tasks", "task_due_day", due_day],
                    created_at=min(task.created_at for task in ordered),
                    updated_at=max(task.updated_at for task in ordered),
                )
            )
        return units

    def _due_day_edges(self, due_days: list[KnowledgeUnit], tasks: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        due_day_ids = {str(unit.metadata["due_date"]): unit.source_id for unit in due_days}
        edges: list[KnowledgeEdge] = []
        for task in tasks:
            due_day = self._due_day(task.metadata.get("due"))
            due_day_id = due_day_ids.get(due_day)
            if not due_day_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=self._edge_id(due_day_id, task.source_id, "task_due_day_contains_task"),
                    from_unit_id=due_day_id,
                    to_unit_id=task.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.GOOGLE_TASKS.value,
                        "relation_type": "task_due_day_contains_task",
                        "due_date": due_day,
                    },
                    created_at=task.created_at,
                )
            )
        return edges

    def _due_day_source_id(self, due_day: str) -> str:
        return f"google_tasks:task_due_day:{due_day}"

    def _due_day(self, value: Any) -> str:
        parsed = self._parse_datetime(value)
        if parsed is not None:
            return parsed.date().isoformat()
        raw = str(value or "").strip()
        if len(raw) >= 10:
            return raw[:10]
        return ""

    def _recurrence_metadata(self, item: dict[str, Any]) -> Any:
        values: dict[str, Any] = {}
        for key in ("recurrence", "repeat", "recurrenceRule", "recurrence_rule"):
            value = self._clean_recurrence_value(item.get(key))
            if value not in (None, "", [], {}):
                values[key] = value
        if not values:
            return None
        if len(values) == 1:
            return next(iter(values.values()))
        return values

    def _clean_recurrence_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {
                str(key): self._clean_recurrence_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
            return {key: item for key, item in cleaned.items() if item not in (None, "", [], {})}
        if isinstance(value, list):
            return [item for item in (self._clean_recurrence_value(item) for item in value) if item not in (None, "", [], {})]
        if isinstance(value, str):
            return value.strip()
        return value

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        raw = str(value).strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (ValueError, TypeError):
            return None
