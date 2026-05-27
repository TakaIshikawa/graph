"""Adapter for Notion database CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class NotionDatabaseCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "notion_database_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["page"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "page" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or max(unit.created_at, unit.updated_at) > sync_at):
                    result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Name", "Title")
        if not title:
            return None
        url = first(row, "URL", "Url")
        created_text = first(row, "Created time", "Created", "Created At")
        edited_text = first(row, "Last edited time", "Last Edited", "Updated")
        created_at = parse_datetime(created_text)
        edited_at = parse_datetime(edited_text)
        tags = split_values(first(row, "Tags", "Tag"))
        now = datetime.now(timezone.utc)
        properties = {key: value for key, value in row.items()}
        metadata = clean_metadata(
            {
                "url": url,
                "title": title,
                "created_time": created_at.isoformat() if created_at else created_text,
                "last_edited_time": edited_at.isoformat() if edited_at else edited_text,
                "tags": tags,
                "properties": properties,
                "source_file": source_file,
            }
        )
        body = first(row, "Content", "Body", "Description", "Text")
        return KnowledgeUnit(
            source_project="notion_database_csv",
            source_id=digest_source_id("notion_database_csv", url or title, created_text or edited_text or index),
            source_entity_type="page",
            title=title,
            content="\n".join(part for part in [title, body, f"URL: {url}" if url else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at or edited_at or now,
            updated_at=edited_at or created_at or now,
        )
