"""Adapter for Things 3 CSV task exports."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class ThingsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "things_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["task", "project", "area", "deadline_bucket"]

    def __init__(self, path: str = "", now: datetime | None = None) -> None:
        self.path = path
        self.now = self._ensure_utc(now) if now else None

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        task_units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.created_at <= sync_at:
                    continue
                task_units.append(unit)

        project_units = self._aggregate_units(task_units, "project") if "project" in allowed_types else []
        area_units = self._aggregate_units(task_units, "area") if "area" in allowed_types else []
        deadline_bucket_units = self._deadline_bucket_units(task_units) if "deadline_bucket" in allowed_types else []
        if "project" in allowed_types:
            result.units.extend(project_units)
        if "area" in allowed_types:
            result.units.extend(area_units)
        if "deadline_bucket" in allowed_types:
            result.units.extend(deadline_bucket_units)
        if "task" in allowed_types:
            result.units.extend(task_units)
        if "project" in allowed_types and "task" in allowed_types:
            result.edges.extend(self._aggregate_edges(project_units, task_units, "project"))
        if "area" in allowed_types and "task" in allowed_types:
            result.edges.extend(self._aggregate_edges(area_units, task_units, "area"))
        if "deadline_bucket" in allowed_types and "task" in allowed_types:
            result.edges.extend(self._deadline_bucket_edges(deadline_bucket_units))
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if not root.is_dir():
            return []
        return sorted(root.glob("*.csv"), key=lambda child: child.name)

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def _unit_from_row(self, row: dict[str, str], source_file: str) -> KnowledgeUnit | None:
        title = self._first(row, "Title", "title", "Task", "Name")
        if not title:
            return None
        notes = self._first(row, "Notes", "notes", "Note")
        created_at = self._parse_datetime(self._first(row, "Creation Date", "Created", "created_at"))
        completed_at = self._parse_datetime(self._first(row, "Completion Date", "Completed", "completed_at"))
        if created_at is None:
            created_at = completed_at or datetime.now(timezone.utc)

        canceled = self._parse_bool(self._first(row, "Canceled", "Cancelled", "canceled"))
        status = self._status(row, completed_at, canceled)
        tags = self._split_tags(self._first(row, "Tags", "tags"))
        metadata = {
            "title": title,
            "notes": notes,
            "area": self._first(row, "Area", "area"),
            "project": self._first(row, "Project", "project"),
            "tags": tags,
            "status": status,
            "creation_date": created_at.isoformat() if created_at else None,
            "start_date": self._datetime_metadata(self._first(row, "Start Date", "When", "start_date")),
            "deadline": self._datetime_metadata(self._first(row, "Deadline", "Due Date", "deadline")),
            "completion_date": completed_at.isoformat() if completed_at else None,
            "canceled": canceled,
            "checklist": self._checklist(self._first(row, "Checklist", "checklist")),
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project=SourceProject.THINGS_CSV,
            source_id=self._source_id(row, title, created_at),
            source_entity_type="task",
            title=title,
            content=self._content(title, notes),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["things", "task", *tags],
            created_at=created_at,
            updated_at=completed_at or created_at,
        )

    def _status(self, row: dict[str, str], completed_at: datetime | None, canceled: bool | None) -> str:
        explicit = self._first(row, "Status", "status")
        if explicit:
            return explicit.lower()
        if canceled:
            return "canceled"
        if completed_at:
            return "completed"
        if self._first(row, "Start Date", "When", "start_date"):
            return "scheduled"
        return "open"

    def _source_id(self, row: dict[str, str], title: str, created_at: datetime) -> str:
        explicit = self._first(row, "UUID", "ID", "id", "uuid")
        identifier = explicit or "|".join([title, created_at.isoformat(), self._first(row, "Project", "project")])
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:24]
        return f"things_csv:{digest}"

    def _aggregate_units(self, tasks: list[KnowledgeUnit], field: str) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for task in tasks:
            name = str(task.metadata.get(field) or "").strip()
            if not name:
                continue
            key = self._aggregate_key(name)
            grouped.setdefault(key, []).append(task)
            names.setdefault(key, name)

        units: list[KnowledgeUnit] = []
        for key, field_tasks in sorted(grouped.items()):
            name = names[key]
            statuses = [str(task.metadata.get("status") or "") for task in field_tasks]
            metadata = {
                "name": name,
                "normalized_name": key,
                "task_count": len(field_tasks),
                "open_count": sum(1 for status in statuses if status == "open"),
                "completed_count": sum(1 for status in statuses if status == "completed"),
                "canceled_count": sum(1 for status in statuses if status == "canceled"),
                "task_source_ids": sorted(task.source_id for task in field_tasks),
                "first_created_at": min(task.created_at for task in field_tasks).isoformat(),
                "latest_updated_at": max(task.updated_at for task in field_tasks).isoformat(),
            }
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.THINGS_CSV,
                    source_id=f"things_csv:{field}:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:24]}",
                    source_entity_type=field,
                    title=name,
                    content=f"Things {field}: {name}\nTasks: {len(field_tasks)}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=["things", field, name],
                    created_at=min(task.created_at for task in field_tasks),
                    updated_at=max(task.updated_at for task in field_tasks),
                )
            )
        return units

    def _aggregate_edges(self, aggregates: list[KnowledgeUnit], tasks: list[KnowledgeUnit], field: str) -> list[KnowledgeEdge]:
        aggregate_ids = {str(unit.metadata.get("normalized_name")): unit.source_id for unit in aggregates}
        edges: list[KnowledgeEdge] = []
        for task in tasks:
            key = self._aggregate_key(str(task.metadata.get(field) or ""))
            aggregate_id = aggregate_ids.get(key)
            if not aggregate_id:
                continue
            digest = hashlib.sha256("|".join((aggregate_id, task.source_id, f"{field}_contains_task")).encode("utf-8")).hexdigest()[:24]
            edges.append(
                KnowledgeEdge(
                    id=f"things-csv-{field}-contains-{digest}",
                    from_unit_id=aggregate_id,
                    to_unit_id=task.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={"source_project": SourceProject.THINGS_CSV.value, "relation_type": f"{field}_contains_task"},
                )
            )
        return edges

    def _deadline_bucket_units(self, tasks: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {bucket: [] for bucket in self._deadline_bucket_names()}
        for task in tasks:
            grouped[self._deadline_bucket(task)].append(task)

        units: list[KnowledgeUnit] = []
        for bucket in self._deadline_bucket_names():
            bucket_tasks = grouped[bucket]
            if not bucket_tasks:
                continue
            statuses = [str(task.metadata.get("status") or "") for task in bucket_tasks]
            deadlines = [
                parsed
                for task in bucket_tasks
                if (parsed := self._parse_datetime(str(task.metadata.get("deadline") or ""))) is not None
            ]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.THINGS_CSV,
                    source_id=f"things_csv:deadline_bucket:{bucket}",
                    source_entity_type="deadline_bucket",
                    title=bucket.replace("_", " ").title(),
                    content=f"Things deadline bucket: {bucket}\nTasks: {len(bucket_tasks)}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "bucket": bucket,
                        "task_count": len(bucket_tasks),
                        "open_count": sum(1 for status in statuses if status not in {"completed", "canceled"}),
                        "completed_count": sum(1 for status in statuses if status == "completed"),
                        "first_deadline": min(deadlines).isoformat() if deadlines else None,
                        "latest_deadline": max(deadlines).isoformat() if deadlines else None,
                        "task_source_ids": sorted(task.source_id for task in bucket_tasks),
                    },
                    tags=["things", "deadline-bucket", bucket],
                    created_at=min(task.created_at for task in bucket_tasks),
                    updated_at=max(task.updated_at for task in bucket_tasks),
                )
            )
        return units

    def _deadline_bucket_edges(self, buckets: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for bucket in buckets:
            for task_source_id in bucket.metadata.get("task_source_ids") or []:
                digest = hashlib.sha256("|".join((bucket.source_id, str(task_source_id), "deadline_bucket_contains_task")).encode("utf-8")).hexdigest()[:24]
                edges.append(
                    KnowledgeEdge(
                        id=f"things-csv-deadline-bucket-contains-{digest}",
                        from_unit_id=bucket.source_id,
                        to_unit_id=str(task_source_id),
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.THINGS_CSV.value,
                            "relation_type": "deadline_bucket_contains_task",
                        },
                    )
                )
        return edges

    def _deadline_bucket(self, task: KnowledgeUnit) -> str:
        deadline = self._parse_datetime(str(task.metadata.get("deadline") or ""))
        if deadline is None:
            return "no_deadline"
        today = self._now().date()
        deadline_date = deadline.date()
        if deadline_date < today:
            return "overdue"
        if deadline_date == today:
            return "today"
        if deadline_date <= today + timedelta(days=7):
            return "upcoming"
        return "later"

    def _deadline_bucket_names(self) -> tuple[str, ...]:
        return ("overdue", "today", "upcoming", "later", "no_deadline")

    def _now(self) -> datetime:
        return self.now or datetime.now(timezone.utc)

    def _aggregate_key(self, name: str) -> str:
        return " ".join(name.casefold().split())

    def _content(self, title: str, notes: str) -> str:
        return f"{title}\n\n{notes}".strip()

    def _checklist(self, value: str) -> Any:
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else value
        except json.JSONDecodeError:
            return [item.strip() for item in value.splitlines() if item.strip()]

    def _split_tags(self, value: str) -> list[str]:
        if not value:
            return []
        normalized = value.replace(";", ",")
        return [tag.strip() for tag in normalized.split(",") if tag.strip()]

    def _first(self, row: dict[str, str], *keys: str) -> str:
        lowered = {key.lower(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.lower())
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _datetime_metadata(self, value: str) -> str | None:
        parsed = self._parse_datetime(value)
        return parsed.isoformat() if parsed else None

    def _parse_bool(self, value: str) -> bool | None:
        if not value:
            return None
        text = value.strip().lower()
        if text in {"true", "t", "yes", "y", "1", "canceled", "cancelled"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
        return None

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        for candidate in (value, f"{value}T00:00:00"):
            try:
                parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
                return self._ensure_utc(parsed)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
