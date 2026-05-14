"""Adapter for YouTube playlist JSON exports."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class YouTubePlaylistsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "youtube_playlists_json"

    @property
    def entity_types(self) -> list[str]:
        return ["playlist", "playlist_video"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else set(self.entity_types)
        if not allowed.intersection(self.entity_types):
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        playlists: list[KnowledgeUnit] = []
        videos: list[KnowledgeUnit] = []
        edges: list[KnowledgeEdge] = []
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._read_playlists(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for playlist_index, playlist in enumerate(records):
                playlist_unit = self._playlist_unit(playlist, path.name, playlist_index)
                if playlist_unit is None:
                    continue
                playlist_videos: list[KnowledgeUnit] = []
                for video_index, video in enumerate(self._videos(playlist)):
                    video_unit = self._video_unit(video, playlist_unit, path.name, video_index)
                    if video_unit is None:
                        continue
                    if sync_at and video_unit.updated_at <= sync_at:
                        continue
                    playlist_videos.append(video_unit)
                    edges.append(self._edge(playlist_unit.source_id, video_unit.source_id))
                playlists.append(playlist_unit)
                videos.extend(playlist_videos)

        if "playlist" in allowed:
            result.units.extend(playlists)
        if "playlist_video" in allowed:
            result.units.extend(videos)
        if {"playlist", "playlist_video"}.issubset(allowed):
            result.edges.extend({edge.id: edge for edge in edges}.values())
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _read_playlists(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if not isinstance(parsed, dict):
            return []
        for key in ("playlists", "items", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = self._nested_playlists(value)
                if nested:
                    return nested
        return [parsed] if first(parsed, "title", "name", "playlistId", "id") else []

    def _nested_playlists(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("playlists", "items", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return []

    def _playlist_unit(self, playlist: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(playlist, "title", "name", "playlistTitle")
        playlist_id = first(playlist, "playlistId", "playlist_id", "id")
        if not (title or playlist_id or self._videos(playlist)):
            return None
        created = parse_datetime(first(playlist, "createdAt", "created_at", "publishedAt", "published_at"))
        updated = parse_datetime(first(playlist, "updatedAt", "updated_at", "modifiedAt", "modified_at")) or created or datetime.now(timezone.utc)
        description = first(playlist, "description", "Description")
        privacy = first(playlist, "privacy", "privacyStatus", "visibility")
        metadata = clean_metadata(
            {
                "playlist_id": playlist_id,
                "title": title,
                "description": description,
                "privacy": privacy,
                "source_file": source_file,
                "record": dict(playlist),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.YOUTUBE_PLAYLISTS_JSON,
            source_id=self._playlist_source_id(playlist_id, title, index),
            source_entity_type="playlist",
            title=title or playlist_id or "YouTube playlist",
            content=self._playlist_content(title or playlist_id or "YouTube playlist", description, privacy),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["youtube", "playlist"],
            created_at=created or updated,
            updated_at=updated,
        )

    def _video_unit(self, video: dict[str, Any], playlist: KnowledgeUnit, source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(video, "title", "name", "videoTitle")
        url = first(video, "url", "videoUrl", "video_url", "link")
        video_id = first(video, "videoId", "video_id", "id") or self._video_id_from_url(url)
        if not (title or video_id or url):
            return None
        added_at = parse_datetime(first(video, "addedAt", "added_at", "timeAdded", "publishedAt", "published_at"))
        updated_at = parse_datetime(first(video, "updatedAt", "updated_at", "modifiedAt", "modified_at")) or added_at or datetime.now(timezone.utc)
        channel = first(video, "channel", "channelTitle", "channel_name", "creator")
        position = self._position(video, index)
        metadata = clean_metadata(
            {
                "playlist_source_id": playlist.source_id,
                "playlist_title": playlist.title,
                "title": title,
                "channel": channel,
                "video_id": video_id,
                "url": url,
                "position": position,
                "added_at": added_at.isoformat() if added_at else None,
                "updated_at": updated_at.isoformat() if updated_at and updated_at != added_at else None,
                "source_file": source_file,
                "record": dict(video),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.YOUTUBE_PLAYLISTS_JSON,
            source_id=self._video_source_id(playlist.source_id, video_id, url, title, position),
            source_entity_type="playlist_video",
            title=title or video_id or url,
            content=self._video_content(title, channel, url, position),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["youtube", "playlist_video"],
            created_at=added_at or updated_at,
            updated_at=updated_at,
        )

    def _videos(self, playlist: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("videos", "items", "entries", "playlistItems"):
            value = playlist.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    def _playlist_source_id(self, playlist_id: str, title: str, index: int) -> str:
        if playlist_id:
            return f"youtube_playlists_json:playlist:{playlist_id}"
        return digest_source_id("youtube_playlists_json:playlist", title, index)

    def _video_source_id(self, playlist_source_id: str, video_id: str, url: str, title: str, position: int) -> str:
        return digest_source_id("youtube_playlists_json:playlist_video", playlist_source_id, video_id or url or title, position)

    def _edge(self, playlist_id: str, video_id: str) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=digest_source_id("youtube-playlists-json-edge", playlist_id, video_id, "playlist_contains_video"),
            from_unit_id=playlist_id,
            to_unit_id=video_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={"source_project": SourceProject.YOUTUBE_PLAYLISTS_JSON.value, "relation_type": "playlist_contains_video"},
        )

    def _video_id_from_url(self, url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if query_id:
            return query_id
        match = re.search(r"/(?:shorts/|embed/)?([A-Za-z0-9_-]{6,})", parsed.path)
        return match.group(1) if match else ""

    def _position(self, video: dict[str, Any], index: int) -> int:
        raw = first(video, "position", "index", "order")
        try:
            return int(raw)
        except ValueError:
            return index

    def _playlist_content(self, title: str, description: str, privacy: str) -> str:
        parts = [title]
        if description:
            parts.append(description)
        if privacy:
            parts.append(f"Privacy: {privacy}")
        return "\n".join(parts)

    def _video_content(self, title: str, channel: str, url: str, position: int) -> str:
        parts = [title] if title else []
        parts.append(f"Position: {position}")
        if channel:
            parts.append(f"Channel: {channel}")
        if url:
            parts.append(f"URL: {url}")
        return "\n".join(parts)
