"""Adapter for Airtable base inventory CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AirtableBasesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "airtable_bases_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["base"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "base" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows, start=2):
                unit = self._unit(row, path.name, index)
                if unit is None or (sync_at and unit.updated_at <= sync_at):
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, source_row: int) -> KnowledgeUnit | None:
        base_id = first(row, "Base ID", "ID", "base_id")
        name = first(row, "Base Name", "Name", "Title")
        workspace = first(row, "Workspace", "Workspace Name")
        if not any([base_id, name, workspace]):
            return None
        created = parse_datetime(first(row, "Created", "Created At", "Created Time"))
        updated = parse_datetime(first(row, "Updated", "Updated At", "Modified", "Last Modified"))
        url = first(row, "URL", "Base URL")
        metadata = clean_metadata({"base_id": base_id, "name": name, "workspace": workspace, "table_count": parse_int(first(row, "Table Count", "Tables")), "collaborator_count": parse_int(first(row, "Collaborator Count", "Collaborators")), "created_at": created.isoformat() if created else first(row, "Created", "Created At"), "updated_at": updated.isoformat() if updated else first(row, "Updated", "Updated At"), "url": url, "source_url": url, "external_url": url, "role": first(row, "Role", "Permission", "Permissions"), "source_file": source_file, "source_row": source_row})
        now = datetime.now(timezone.utc)
        title = name or base_id or "Airtable base"
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{base_id}" if base_id else digest_source_id(self.name, name, workspace), source_entity_type="base", title=title, content=self._content(title, metadata), content_type=ContentType.METADATA, metadata=metadata, tags=list(dict.fromkeys(tag for tag in ["airtable", "base", workspace] if tag)), created_at=created or now, updated_at=updated or created or now)

    def _content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [title]
        for key, label in (("workspace", "Workspace"), ("table_count", "Tables"), ("collaborator_count", "Collaborators"), ("url", "URL")):
            if key in metadata:
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)
