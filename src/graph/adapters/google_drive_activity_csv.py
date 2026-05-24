"""Adapter for Google Drive activity CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GoogleDriveActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_drive_activity_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["activity"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "activity" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None or (sync_at and ("timestamp" not in unit.metadata or unit.updated_at <= sync_at)):
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        time_text = first(row, "Time", "Date", "Timestamp", "Activity Time", "Event Time")
        timestamp = parse_datetime(time_text)
        file_name = first(row, "File Name", "Filename", "Title", "Name", "Document Title")
        actor = first(row, "Actor", "User", "Email", "User Email")
        action = first(row, "Action", "Event", "Activity", "Event Type")
        owner = first(row, "Owner", "File Owner", "Document Owner")
        mime_type = first(row, "MIME Type", "Mime Type", "Content Type")
        url = first(row, "URL", "Url", "Link", "Document URL")
        file_id = first(row, "File ID", "File Id", "Document ID", "ID")
        if not any([time_text, file_name, actor, action, owner, mime_type, url, file_id]):
            return None
        now = datetime.now(timezone.utc)
        occurred_at = timestamp or now
        metadata = clean_metadata(
            {
                "timestamp": timestamp.isoformat() if timestamp else time_text,
                "file_name": file_name,
                "file_id": file_id,
                "actor": actor,
                "action": action,
                "owner": owner,
                "mime_type": mime_type,
                "url": url,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="google_drive_activity_csv",
            source_id=f"google_drive_activity_csv:{file_id}:{time_text}:{action}:{actor}" if file_id else digest_source_id("google_drive_activity_csv", time_text, file_name, actor, action, owner, mime_type, url, index),
            source_entity_type="activity",
            title=self._title(action, file_name, actor),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["google-drive", "activity", action, actor, mime_type] if tag)),
            created_at=occurred_at,
            updated_at=occurred_at,
        )

    def _title(self, action: str, file_name: str, actor: str) -> str:
        base = action or "Google Drive activity"
        if file_name:
            base = f"{base}: {file_name}"
        if actor:
            base = f"{base} by {actor}"
        return base

    def _content(self, metadata: dict[str, Any]) -> str:
        labels = [("Action", "action"), ("Actor", "actor"), ("File", "file_name"), ("Owner", "owner"), ("MIME Type", "mime_type"), ("URL", "url")]
        return "\n".join(f"{label}: {metadata[key]}" for label, key in labels if key in metadata)
