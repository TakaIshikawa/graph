"""Adapter for Zoom cloud recording CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_duration_seconds, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class ZoomRecordingsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "zoom_recordings_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["recording"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "recording" not in entity_types:
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
        meeting_id = first(row, "Meeting ID", "Meeting Id", "ID")
        uuid = first(row, "UUID", "Meeting UUID")
        topic = first(row, "Topic", "Meeting Topic")
        url = first(row, "Share URL", "Recording URL", "Download URL", "URL")
        if not any([meeting_id, uuid, topic, url]):
            return None
        start = parse_datetime(first(row, "Start Time", "Start", "Recording Start"))
        duration = parse_duration_seconds(first(row, "Duration", "Duration Minutes"))
        host = first(row, "Host", "Host Email", "Host Name")
        recording_type = first(row, "Recording Type", "File Type", "Type")
        transcript_url = first(row, "Transcript URL", "Audio Transcript URL")
        file_size = parse_int(first(row, "File Size", "Size", "File Size Bytes"))
        metadata = clean_metadata(
            {
                "meeting_id": meeting_id,
                "uuid": uuid,
                "topic": topic,
                "host": host,
                "start_time": start.isoformat() if start else "",
                "duration_seconds": duration,
                "recording_type": recording_type,
                "share_url": url,
                "source_url": url,
                "transcript_url": transcript_url,
                "file_size": file_size,
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.ZOOM_RECORDINGS_CSV,
            source_id=digest_source_id("zoom_recordings_csv", uuid or meeting_id or topic, url, index if not (uuid or meeting_id) else ""),
            source_entity_type="recording",
            title=topic or meeting_id or "Zoom recording",
            content="\n".join(part for part in [topic, f"Host: {host}" if host else "", f"Recording: {url}" if url else "", f"Transcript: {transcript_url}" if transcript_url else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["zoom", "recording"],
            created_at=start or now,
            updated_at=start or now,
        )
