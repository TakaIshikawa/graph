"""Adapter for Notion page/database CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class NotionPagesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "notion_pages_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["page"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "page" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows, start=1):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _read_rows(self, path: Any) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            return [{str(key).strip(): "" if value is None else str(value).strip() for key, value in row.items() if key is not None} for row in reader]

    def _unit(self, row: dict[str, Any], source_file: str, row_number: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Name", "Page", "Page Title")
        url = first(row, "URL", "Url", "Link", "Page URL", "Public URL")
        if not title and not url:
            return None
        created_text = first(row, "Created time", "Created", "Created At")
        edited_text = first(row, "Last edited time", "Last Edited", "Updated", "Updated At")
        created_at = parse_datetime(created_text)
        updated_at = parse_datetime(edited_text) or created_at
        tags = split_values(first(row, "Tags", "Tag", "Multi-select", "Multi Select", "Labels"))
        status = first(row, "Status", "State")
        parent = first(row, "Parent", "Parent page", "Parent Page")
        database = first(row, "Database", "Database Name", "Database ID")
        metadata = clean_metadata(
            {
                "title": title,
                "url": url,
                "created_time": created_at.isoformat() if created_at else created_text,
                "last_edited_time": updated_at.isoformat() if updated_at else edited_text,
                "tags": tags,
                "status": status,
                "parent": parent,
                "database": database,
                "source_file": source_file,
                "row_number": row_number,
            }
        )
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project="notion_pages_csv",
            source_id=f"notion_pages_csv:{url}" if url else digest_source_id("notion_pages_csv", title, parent, database, row_number),
            source_entity_type="page",
            title=title or url,
            content=self._content(title, url, status, parent, database, tags),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(["notion", "page", *tags, status] if status else ["notion", "page", *tags])),
            created_at=created_at or now,
            updated_at=updated_at or created_at or now,
        )

    def _content(self, title: str, url: str, status: str, parent: str, database: str, tags: list[str]) -> str:
        parts = [title, f"URL: {url}" if url else "", f"Status: {status}" if status else "", f"Parent: {parent}" if parent else "", f"Database: {database}" if database else "", f"Tags: {', '.join(tags)}" if tags else ""]
        return "\n".join(part for part in parts if part)
