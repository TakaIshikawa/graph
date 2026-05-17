"""Adapter for Clockify time entries CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import (
    clean_metadata,
    digest_source_id,
    ensure_utc,
    first,
    iter_paths,
    parse_datetime,
    parse_duration_seconds,
    parse_money,
    read_csv_rows,
    split_values,
)
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class ClockifyTimeEntriesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "clockify_time_entries_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["time_entry", "project", "client"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types) if entity_types is not None else set(self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        entry_units: list[KnowledgeUnit] = []
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._entry_unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                entry_units.append(unit)

        project_units = self._project_units(entry_units)
        client_units = self._client_units(entry_units)
        if "time_entry" in allowed_types:
            result.units.extend(entry_units)
        if "project" in allowed_types:
            result.units.extend(project_units)
        if "client" in allowed_types:
            result.units.extend(client_units)
        if {"project", "time_entry"}.issubset(allowed_types):
            result.edges.extend(self._project_edges(project_units, entry_units))
        if {"client", "time_entry"}.issubset(allowed_types):
            result.edges.extend(self._client_edges(client_units, entry_units))

        result.units.sort(key=lambda unit: (unit.source_entity_type, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _entry_unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        entry_id = first(row, "Entry ID", "Time Entry ID", "ID", "Id")
        project = first(row, "Project", "Project Name")
        client = first(row, "Client", "Client Name")
        task = first(row, "Task", "Task Name")
        description = first(row, "Description", "Time Entry Description")
        user = first(row, "User", "User Name", "Name")
        start_at = self._date_time(row, "Start")
        end_at = self._date_time(row, "End")
        duration_seconds = parse_duration_seconds(first(row, "Duration", "Duration (h)", "Hours"))
        if duration_seconds is None and start_at and end_at:
            duration_seconds = max(0, int((end_at - start_at).total_seconds()))
        if not any([entry_id, start_at, end_at, duration_seconds is not None, project, client, task, description, user]):
            return None

        now = datetime.now(timezone.utc)
        created_at = start_at or end_at or now
        metadata = clean_metadata(
            {
                "entry_id": entry_id,
                "project": project,
                "client": client,
                "task": task,
                "description": description,
                "user": user,
                "start_at": start_at.isoformat() if start_at else "",
                "end_at": end_at.isoformat() if end_at else "",
                "duration_seconds": duration_seconds,
                "billable": self._bool(first(row, "Billable", "Is Billable")),
                "tags": split_values(first(row, "Tags", "Tag")),
                "hourly_rate": parse_money(first(row, "Hourly Rate", "Rate")),
                "source_file": source_file,
            }
        )
        title_parts = [part for part in (project, task, description) if part]
        title = " - ".join(title_parts) or f"Clockify entry {metadata.get('start_at', index)}"
        return KnowledgeUnit(
            source_project="clockify_time_entries_csv",
            source_id=self._entry_source_id(entry_id, start_at, end_at, project, task, description, index),
            source_entity_type="time_entry",
            title=title,
            content=self._entry_content(title, metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["clockify", "time_entry", project, client, task] if tag)),
            created_at=created_at,
            updated_at=end_at or created_at,
        )

    def _date_time(self, row: dict[str, Any], prefix: str) -> datetime | None:
        combined = first(row, f"{prefix}", f"{prefix} Time")
        if combined:
            parsed = self._parse_datetime(combined)
            if parsed:
                return parsed
        date_text = first(row, f"{prefix} Date", f"{prefix} date")
        time_text = first(row, f"{prefix} Time", f"{prefix} time")
        return self._parse_datetime(" ".join(part for part in (date_text, time_text) if part))

    def _parse_datetime(self, value: Any) -> datetime | None:
        parsed = parse_datetime(value)
        if parsed:
            return parsed
        text = "" if value is None else str(value).strip()
        for fmt in (
            "%m/%d/%Y %I:%M %p",
            "%m/%d/%Y %I:%M:%S %p",
            "%Y-%m-%d %I:%M %p",
            "%Y-%m-%d %I:%M:%S %p",
        ):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _entry_source_id(
        self,
        entry_id: str,
        start_at: datetime | None,
        end_at: datetime | None,
        project: str,
        task: str,
        description: str,
        index: int,
    ) -> str:
        if entry_id:
            return digest_source_id("clockify_time_entries_csv", entry_id)
        return digest_source_id(
            "clockify_time_entries_csv",
            start_at.isoformat() if start_at else "",
            end_at.isoformat() if end_at else "",
            project,
            task,
            description,
            index if not any([start_at, end_at, project, task, description]) else "",
        )

    def _project_units(self, entries: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for entry in entries:
            project = str(entry.metadata.get("project") or "").strip()
            if not project:
                continue
            key = project.casefold()
            names.setdefault(key, project)
            grouped.setdefault(key, []).append(entry)

        units: list[KnowledgeUnit] = []
        for key, project_entries in sorted(grouped.items()):
            name = names[key]
            metadata = self._aggregate_metadata(project_entries, {"project": name})
            units.append(
                KnowledgeUnit(
                    source_project="clockify_time_entries_csv",
                    source_id=digest_source_id("clockify_time_entries_csv_project", key),
                    source_entity_type="project",
                    title=name,
                    content=f"Clockify project: {name}\nEntries: {len(project_entries)}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=["clockify", "project", name],
                    created_at=min(entry.created_at for entry in project_entries),
                    updated_at=max(entry.updated_at for entry in project_entries),
                )
            )
        return units

    def _client_units(self, entries: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for entry in entries:
            client = str(entry.metadata.get("client") or "").strip()
            if not client:
                continue
            key = client.casefold()
            names.setdefault(key, client)
            grouped.setdefault(key, []).append(entry)

        units: list[KnowledgeUnit] = []
        for key, client_entries in sorted(grouped.items()):
            name = names[key]
            metadata = self._aggregate_metadata(client_entries, {"client": name})
            units.append(
                KnowledgeUnit(
                    source_project="clockify_time_entries_csv",
                    source_id=digest_source_id("clockify_time_entries_csv_client", key),
                    source_entity_type="client",
                    title=name,
                    content=f"Clockify client: {name}\nEntries: {len(client_entries)}",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=["clockify", "client", name],
                    created_at=min(entry.created_at for entry in client_entries),
                    updated_at=max(entry.updated_at for entry in client_entries),
                )
            )
        return units

    def _aggregate_metadata(self, entries: list[KnowledgeUnit], base: dict[str, Any]) -> dict[str, Any]:
        durations = [value for entry in entries if (value := entry.metadata.get("duration_seconds")) is not None]
        billable_durations = [
            value
            for entry in entries
            if entry.metadata.get("billable") is True and (value := entry.metadata.get("duration_seconds")) is not None
        ]
        metadata = {
            **base,
            "time_entry_count": len(entries),
            "total_duration_seconds": sum(durations),
            "billable_duration_seconds": sum(billable_durations),
            "projects": sorted({str(entry.metadata.get("project")) for entry in entries if entry.metadata.get("project")}),
            "clients": sorted({str(entry.metadata.get("client")) for entry in entries if entry.metadata.get("client")}),
            "tasks": sorted({str(entry.metadata.get("task")) for entry in entries if entry.metadata.get("task")}),
            "users": sorted({str(entry.metadata.get("user")) for entry in entries if entry.metadata.get("user")}),
            "tags": sorted({tag for entry in entries for tag in entry.metadata.get("tags", [])}),
            "source_files": sorted({str(entry.metadata.get("source_file")) for entry in entries if entry.metadata.get("source_file")}),
            "time_entry_source_ids": sorted(entry.source_id for entry in entries),
            "first_entry_at": min(entry.created_at for entry in entries).isoformat(),
            "last_entry_at": max(entry.updated_at for entry in entries).isoformat(),
        }
        return clean_metadata(metadata)

    def _project_edges(self, projects: list[KnowledgeUnit], entries: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        project_ids = {str(project.metadata.get("project") or "").casefold(): project.source_id for project in projects}
        edges: list[KnowledgeEdge] = []
        for entry in entries:
            project_id = project_ids.get(str(entry.metadata.get("project") or "").casefold())
            if not project_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=digest_source_id("clockify_time_entries_csv_project_edge", project_id, entry.source_id),
                    from_unit_id=project_id,
                    to_unit_id=entry.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={"relation_type": "project_contains_time_entry", "project": entry.metadata.get("project")},
                )
            )
        return edges

    def _client_edges(self, clients: list[KnowledgeUnit], entries: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        client_ids = {str(client.metadata.get("client") or "").casefold(): client.source_id for client in clients}
        edges: list[KnowledgeEdge] = []
        for entry in entries:
            client_id = client_ids.get(str(entry.metadata.get("client") or "").casefold())
            if not client_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=digest_source_id("clockify_time_entries_csv_client_edge", client_id, entry.source_id),
                    from_unit_id=client_id,
                    to_unit_id=entry.source_id,
                    relation=EdgeRelation.RELATES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={"relation_type": "client_time_entry", "client": entry.metadata.get("client")},
                )
            )
        return edges

    def _bool(self, value: Any) -> bool | None:
        text = "" if value is None else str(value).strip().casefold()
        if text in {"true", "yes", "y", "1", "billable"}:
            return True
        if text in {"false", "no", "n", "0", "non-billable", "non billable"}:
            return False
        return None

    def _entry_content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (
            ("client", "Client"),
            ("project", "Project"),
            ("task", "Task"),
            ("description", "Description"),
            ("user", "User"),
            ("start_at", "Start"),
            ("end_at", "End"),
            ("duration_seconds", "Duration seconds"),
            ("billable", "Billable"),
            ("tags", "Tags"),
            ("hourly_rate", "Hourly rate"),
        ):
            if metadata.get(key) not in ("", None, []):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
