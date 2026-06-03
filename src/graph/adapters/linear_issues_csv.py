"""Adapter for Linear issue CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class LinearIssuesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "linear_issues_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["issue"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "issue" not in set(entity_types or self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _read_rows(self, path: Any) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        identifier = first(row, "Identifier", "Issue ID", "ID", "Key")
        title = first(row, "Title", "Name", "Summary")
        description = first(row, "Description")
        if not identifier and not title:
            return None
        created = parse_datetime(first(row, "Created", "Created At", "Created Date"))
        updated = parse_datetime(first(row, "Updated", "Updated At", "Modified", "Modified At")) or created
        archived_at = parse_datetime(first(row, "Archived At", "Archived"))
        completed_at = parse_datetime(first(row, "Completed At", "Completed"))
        labels = split_values(first(row, "Labels", "Label"))
        status = first(row, "Status", "State", "Workflow State")
        metadata = {
            "identifier": identifier,
            "title": title,
            "description": description,
            "status": status,
            "priority": first(row, "Priority"),
            "assignee": first(row, "Assignee", "Assigned To"),
            "team": first(row, "Team", "Team Name"),
            "project": first(row, "Project", "Project Name"),
            "labels": labels,
            "created_at": created.isoformat() if created else first(row, "Created", "Created At"),
            "updated_at": updated.isoformat() if updated else first(row, "Updated", "Updated At"),
            "completed_at": completed_at.isoformat() if completed_at else first(row, "Completed At", "Completed"),
            "archived_at": archived_at.isoformat() if archived_at else first(row, "Archived At", "Archived"),
            "url": first(row, "URL", "Link", "Issue URL"),
            "source_file": source_file,
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.LINEAR_ISSUES_CSV,
            source_id=f"linear_issues_csv:{identifier}" if identifier else digest_source_id("linear_issues_csv", title, description, created),
            source_entity_type="issue",
            title=title or identifier,
            content=self._content(title or identifier, description, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["linear", "issue", status, *labels] if tag)),
            created_at=created or now,
            updated_at=updated or completed_at or archived_at or created or now,
        )

    def _content(self, title: str, description: str, metadata: dict[str, Any]) -> str:
        parts = [title, description]
        for key, label in (("status", "Status"), ("priority", "Priority"), ("assignee", "Assignee"), ("team", "Team"), ("project", "Project"), ("url", "URL")):
            if metadata.get(key):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(part for part in parts if part)
