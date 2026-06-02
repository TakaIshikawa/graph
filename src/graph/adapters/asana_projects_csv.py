"""Adapter for Asana project CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class AsanaProjectsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "asana_projects_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["project"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "project" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        name = first(row, "Name", "Project Name")
        notes = first(row, "Notes", "Description")
        project_id = first(row, "GID", "Project GID", "ID", "Project ID")
        if not any([project_id, name, notes]):
            return None
        created = parse_datetime(first(row, "Created At", "Created"))
        modified = parse_datetime(first(row, "Modified At", "Updated At", "Modified")) or created
        archived_text = first(row, "Archived")
        url = first(row, "Permalink", "URL")
        owner = first(row, "Owner", "Project Owner")
        team = first(row, "Team")
        workspace = first(row, "Workspace")
        due_on = first(row, "Due On", "Due Date")
        metadata = clean_metadata(
            {
                "project_id": project_id,
                "owner": owner,
                "team": team,
                "workspace": workspace,
                "archived": archived_text.casefold() in {"true", "yes", "1", "archived"} if archived_text else "",
                "created_at": created.isoformat() if created else "",
                "modified_at": modified.isoformat() if modified else "",
                "due_on": due_on,
                "permalink": url,
                "source_url": url,
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.ASANA_PROJECTS_CSV,
            source_id=digest_source_id("asana_projects_csv", project_id or name, index if not project_id else ""),
            source_entity_type="project",
            title=name or project_id or "Asana project",
            content="\n".join(part for part in [name, notes, f"Owner: {owner}" if owner else "", f"URL: {url}" if url else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["asana", "project"],
            created_at=created or modified or now,
            updated_at=modified or created or now,
        )
