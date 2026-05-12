"""Adapter for Asana task CSV exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class AsanaTasksCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "asana_tasks_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["task"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types and "task" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        by_task_id: dict[str, KnowledgeUnit] = {}
        parent_refs: list[tuple[str, str]] = []
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
                task_id = str(unit.metadata.get("task_id") or "")
                if task_id:
                    by_task_id[task_id] = unit
                parent_id = str(unit.metadata.get("parent_task_id") or "")
                if task_id and parent_id:
                    parent_refs.append((parent_id, task_id))
        for parent_id, child_id in parent_refs:
            parent = by_task_id.get(parent_id)
            child = by_task_id.get(child_id)
            if parent and child:
                result.edges.append(self._edge(parent.source_id, child.source_id))
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges = sorted({edge.id: edge for edge in result.edges}.values(), key=lambda edge: edge.id)
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        task_id = first(row, "Task ID", "ID", "Task Id", "gid")
        name = first(row, "Name", "Task Name", "Title")
        notes = first(row, "Notes", "Description")
        if not task_id and not name and not notes:
            return None
        created = parse_datetime(first(row, "Created At", "Created"))
        modified = parse_datetime(first(row, "Modified At", "Last Modified", "Updated At")) or created
        due = parse_datetime(first(row, "Due Date", "Due On", "Due At"))
        completed = parse_datetime(first(row, "Completed At", "Completion Date"))
        projects = split_values(first(row, "Projects", "Project"))
        tags = split_values(first(row, "Tags", "Tag"))
        status = "completed" if completed else first(row, "Status", "Completed") or "open"
        metadata = {
            "task_id": task_id,
            "name": name,
            "notes": notes,
            "assignee": first(row, "Assignee", "Assigned To"),
            "projects": projects,
            "tags": tags,
            "status": status.casefold() if status else "",
            "created_at": created.isoformat() if created else first(row, "Created At", "Created"),
            "modified_at": modified.isoformat() if modified else first(row, "Modified At", "Updated At"),
            "due_date": due.isoformat() if due else first(row, "Due Date", "Due On", "Due At"),
            "completed_at": completed.isoformat() if completed else first(row, "Completed At", "Completion Date"),
            "parent_task_id": first(row, "Parent Task ID", "Parent ID", "Parent"),
            "task_url": first(row, "Task URL", "URL", "Link"),
            "source_file": source_file,
        }
        now = datetime.now(timezone.utc)
        graph_tags = list(dict.fromkeys(["asana", "task", *projects, *tags, metadata["status"]]))
        title = name or f"Asana task {task_id}"
        return KnowledgeUnit(
            source_project=SourceProject.ASANA_TASKS_CSV,
            source_id=f"asana_tasks_csv:{task_id}" if task_id else digest_source_id("asana_tasks_csv", title, notes, created),
            source_entity_type="task",
            title=title,
            content=self._content(title, notes, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=clean_metadata(metadata),
            tags=[tag for tag in graph_tags if tag],
            created_at=created or modified or due or now,
            updated_at=modified or completed or created or due or now,
        )

    def _edge(self, parent_source_id: str, child_source_id: str) -> KnowledgeEdge:
        raw = f"{parent_source_id}|contains|{child_source_id}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(
            id=f"asana_tasks_csv:contains:{digest}",
            from_unit_id=parent_source_id,
            to_unit_id=child_source_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={"source_project": SourceProject.ASANA_TASKS_CSV.value, "relation_type": "parent_task"},
        )

    def _content(self, title: str, notes: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        if notes:
            parts.append(notes)
        for key, label in (("assignee", "Assignee"), ("status", "Status"), ("due_date", "Due"), ("task_url", "URL")):
            if metadata.get(key):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
