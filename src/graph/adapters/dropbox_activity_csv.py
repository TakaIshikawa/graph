"""Adapter for Dropbox activity CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class DropboxActivityCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "dropbox_activity_csv"

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
        time_text = first(row, "Date", "Time", "Timestamp", "Event Time")
        timestamp = parse_datetime(time_text)
        event = first(row, "Event", "Action", "Event Type", "Activity")
        path = first(row, "Path", "File Path", "File", "Item")
        actor = first(row, "Actor", "User", "Email")
        device = first(row, "Device", "Device Name")
        ip_address = first(row, "IP Address", "IP")
        shared_folder = first(row, "Shared Folder", "Shared Folder Name")
        if not any([time_text, event, path, actor, device, ip_address, shared_folder]):
            return None
        now = datetime.now(timezone.utc)
        occurred_at = timestamp or now
        metadata = clean_metadata(
            {
                "timestamp": timestamp.isoformat() if timestamp else time_text,
                "event": event,
                "path": path,
                "actor": actor,
                "device": device,
                "ip_address": ip_address,
                "shared_folder": shared_folder,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="dropbox_activity_csv",
            source_id=digest_source_id("dropbox_activity_csv", time_text, event, path, actor, device, ip_address, shared_folder, index),
            source_entity_type="activity",
            title=self._title(event, path, actor),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["dropbox", "activity", event, actor, shared_folder] if tag)),
            created_at=occurred_at,
            updated_at=occurred_at,
        )

    def _title(self, event: str, path: str, actor: str) -> str:
        title = event or "Dropbox activity"
        if path:
            title = f"{title}: {path}"
        if actor:
            title = f"{title} by {actor}"
        return title

    def _content(self, metadata: dict[str, Any]) -> str:
        labels = [("Event", "event"), ("Path", "path"), ("Actor", "actor"), ("Device", "device"), ("IP Address", "ip_address"), ("Shared Folder", "shared_folder")]
        return "\n".join(f"{label}: {metadata[key]}" for label, key in labels if key in metadata)
