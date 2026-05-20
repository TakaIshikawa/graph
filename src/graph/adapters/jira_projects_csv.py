"""Adapter for Jira project inventory CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class JiraProjectsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "jira_projects_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["project"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else set(self.entity_types)
        if "project" not in allowed:
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        units: dict[str, KnowledgeUnit] = {}
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units[unit.source_id] = unit

        result.units.extend(sorted(units.values(), key=lambda unit: (unit.updated_at, unit.source_id)))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        project_key = first(row, "Project key", "Key", "Project Key", "project_key")
        name = first(row, "Name", "Project name", "Project Name", "project_name")
        project_type = first(row, "Project type", "Type", "Template", "Project Type", "project_type")
        category = first(row, "Category", "Project category", "Project Category", "project_category")
        lead = first(row, "Lead", "Project lead", "Owner", "Project Lead", "project_lead")
        url = first(row, "URL", "Url", "Project URL", "Project Url", "Link")
        archived = self._parse_bool(first(row, "Archived", "Is archived", "Archived?", "Status"))
        created_text = first(row, "Created", "Created at", "Created date", "Created At", "created_at")
        updated_text = first(row, "Updated", "Updated at", "Updated date", "Last updated", "Updated At", "updated_at")
        created = parse_datetime(created_text)
        updated = parse_datetime(updated_text) or created
        description = first(row, "Description", "Project description", "Details", "Notes")
        if not any([project_key, name, project_type, category, lead, url, created_text, updated_text, description]):
            return None

        event_at = updated or created or datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "project_key": project_key,
                "name": name,
                "project_type": project_type,
                "category": category,
                "lead": lead,
                "url": url,
                "source_url": url,
                "archived": archived,
                "created_at": created.isoformat() if created else created_text,
                "updated_at": updated.isoformat() if updated else updated_text,
                "description": description,
                "source_file": source_file,
                "source_row": index + 2,
            }
        )
        title = name or project_key or "Untitled Jira project"
        return KnowledgeUnit(
            source_project="jira_projects_csv",
            source_id=self._source_id(project_key, name, url, index),
            source_entity_type="project",
            title=title,
            content=self._content(title, metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["jira", "project", project_key, project_type, category] if tag)),
            created_at=created or event_at,
            updated_at=event_at,
        )

    def _source_id(self, project_key: str, name: str, url: str, index: int) -> str:
        if project_key:
            return f"jira_projects_csv:{project_key}"
        return digest_source_id("jira_projects_csv", name, url, index)

    def _parse_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if value in (None, ""):
            return None
        text = str(value).strip().casefold()
        if text in {"true", "yes", "y", "1", "archived", "inactive"}:
            return True
        if text in {"false", "no", "n", "0", "active", "current"}:
            return False
        return None

    def _content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for label, key in (
            ("Key", "project_key"),
            ("Type", "project_type"),
            ("Category", "category"),
            ("Lead", "lead"),
            ("Archived", "archived"),
            ("Created", "created_at"),
            ("Updated", "updated_at"),
            ("URL", "url"),
            ("Description", "description"),
        ):
            if key in metadata:
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
