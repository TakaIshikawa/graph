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
        return ["task", "task_list"]

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

        for file_path in files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue

            if not isinstance(data, dict):
                continue

            list_title = data.get("title") or file_path.stem
            list_id_raw = f"list:{list_title}"
            list_digest = hashlib.sha1(list_id_raw.encode("utf-8")).hexdigest()[:16]
            list_source_id = f"google_tasks:task_list:{list_digest}"

            # Create task_list unit
            if not allowed_types or "task_list" in allowed_types:
                unit = KnowledgeUnit(
                    source_project=SourceProject.GOOGLE_TASKS,
                    source_id=list_source_id,
                    source_entity_type="task_list",
                    title=list_title,
                    content=list_title,
                    content_type=ContentType.ARTIFACT,
                    metadata={"list_title": list_title},
                    tags=[],
                )
                result.units.append(unit)

            # Process tasks
            items = data.get("items") or []
            task_id_map: dict[str, str] = {}  # google task id -> source_id

            for item in items:
                if not isinstance(item, dict):
                    continue
                if allowed_types and "task" not in allowed_types:
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
                result.units.append(unit)

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

        result.units.sort(key=lambda u: (u.source_entity_type, u.source_id))
        result.edges.sort(key=lambda e: (e.from_unit_id, e.to_unit_id))
        return result

    def _json_paths(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root]
        return sorted(p for p in root.rglob("*.json") if p.is_file())

    def _edge_id(self, from_id: str, to_id: str, relation_type: str) -> str:
        raw = "|".join([SourceProject.GOOGLE_TASKS.value, relation_type, from_id, to_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"google_tasks-{relation_type}-{digest}"

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
