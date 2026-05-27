"""Adapter for Google Drive file inventory CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GoogleDriveFilesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_drive_files_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["file"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "file" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        name = first(row, "Name", "Title")
        if not name:
            return None
        file_id = first(row, "ID", "File ID")
        url = first(row, "URL", "Web View Link", "Link")
        mime_type = first(row, "MIME Type", "Mime Type")
        owner = first(row, "Owner", "Owners")
        created_text = first(row, "Created Time", "Created")
        modified_text = first(row, "Modified Time", "Modified")
        created_at = parse_datetime(created_text)
        modified_at = parse_datetime(modified_text)
        size = parse_int(first(row, "Size", "Size Bytes"))
        starred = first(row, "Starred")
        description = first(row, "Description")
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "file_id": file_id,
                "url": url,
                "mime_type": mime_type,
                "owner": owner,
                "created_time": created_at.isoformat() if created_at else created_text,
                "modified_time": modified_at.isoformat() if modified_at else modified_text,
                "size": size,
                "starred": starred.casefold() in {"true", "yes", "1", "starred"} if starred else "",
                "description": description,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="google_drive_files_csv",
            source_id=digest_source_id("google_drive_files_csv", file_id or url or name, "" if file_id else index),
            source_entity_type="file",
            title=name,
            content="\n".join(part for part in [name, description, f"MIME type: {mime_type}" if mime_type else "", f"URL: {url}" if url else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            created_at=created_at or modified_at or now,
            updated_at=modified_at or created_at or now,
        )
