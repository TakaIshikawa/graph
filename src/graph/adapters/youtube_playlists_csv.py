"""Adapter for YouTube playlist CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class YoutubePlaylistsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "youtube_playlists_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["playlist_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "playlist_item" not in entity_types:
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
        playlist_id = first(row, "Playlist ID", "Playlist Id")
        playlist_name = first(row, "Playlist Name", "Playlist")
        video_id = first(row, "Video ID", "Video Id")
        video_title = first(row, "Video Title", "Title")
        if not any([playlist_id, playlist_name, video_id, video_title]):
            return None
        added_at = parse_datetime(first(row, "Added At", "Date Added", "Added"))
        url = first(row, "Video URL", "URL")
        channel = first(row, "Channel", "Channel Title", "Channel Name")
        description = first(row, "Description")
        position = parse_int(first(row, "Position", "Index"))
        status = first(row, "Privacy", "Status")
        metadata = clean_metadata(
            {
                "playlist_id": playlist_id,
                "playlist_name": playlist_name,
                "video_id": video_id,
                "video_title": video_title,
                "channel": channel,
                "video_url": url,
                "source_url": url,
                "added_at": added_at.isoformat() if added_at else "",
                "position": position,
                "description": description,
                "status": status,
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        title = video_title or f"{playlist_name} playlist item"
        return KnowledgeUnit(
            source_project=SourceProject.YOUTUBE_PLAYLISTS_CSV,
            source_id=digest_source_id("youtube_playlists_csv", playlist_id or playlist_name, video_id or video_title or url, index if not (playlist_id and video_id) else ""),
            source_entity_type="playlist_item",
            title=title,
            content="\n".join(part for part in [title, f"Playlist: {playlist_name}" if playlist_name else "", description, f"Channel: {channel}" if channel else "", f"URL: {url}" if url else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["youtube", "playlist_item"],
            created_at=added_at or now,
            updated_at=added_at or now,
        )
