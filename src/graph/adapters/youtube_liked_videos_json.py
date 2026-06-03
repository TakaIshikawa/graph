"""Adapter for YouTube liked videos JSON exports."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class YouTubeLikedVideosJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "youtube_liked_videos_json"

    @property
    def entity_types(self) -> list[str]:
        return ["liked_video"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "liked_video" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._records(json.loads(path.read_text(encoding="utf-8-sig")))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit(record, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            records: list[dict[str, Any]] = []
            for item in value:
                records.extend(self._records(item))
            return records
        if not isinstance(value, dict):
            return []
        renderer = value.get("videoRenderer") or value.get("compactVideoRenderer") or value.get("gridVideoRenderer")
        if isinstance(renderer, dict):
            return [renderer]
        if self._looks_like_video(value):
            return [value]
        records = []
        for key in ("likedVideos", "liked_videos", "videos", "items", "entries", "contents", "data"):
            nested = value.get(key)
            if isinstance(nested, (list, dict)):
                records.extend(self._records(nested))
        return records

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = self._runs_text(record.get("title")) or first(record, "title", "name", "videoTitle")
        url = first(record, "url", "videoUrl", "video_url", "link", "href")
        video_id = first(record, "videoId", "video_id", "id") or self._video_id_from_url(url)
        if not url and video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
        channel = first(record, "channel", "channelTitle", "channel_name", "ownerChannelName") or self._runs_text(record.get("ownerText"))
        description = self._runs_text(record.get("descriptionSnippet")) or first(record, "description", "descriptionSnippet", "snippet")
        liked_at = parse_datetime(first(record, "likedAt", "liked_at", "time", "timestamp", "date"))
        updated_at = liked_at or parse_datetime(first(record, "updatedAt", "updated_at")) or datetime.now(timezone.utc)
        if not any((title, url, video_id)):
            return None
        metadata = clean_metadata(
            {
                "video_id": video_id,
                "url": url,
                "title": title,
                "channel": channel,
                "description": description,
                "liked_at": liked_at.isoformat() if liked_at else None,
                "source_file": source_file,
                "record": dict(record),
            }
        )
        return KnowledgeUnit(
            source_project=self.name,
            source_id=f"{self.name}:video:{video_id}" if video_id else digest_source_id(f"{self.name}:video", url or title, index),
            source_entity_type="liked_video",
            title=title or video_id or url,
            content=self._content(title, channel, url, description),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["youtube", "liked_video"],
            created_at=liked_at or updated_at,
            updated_at=updated_at,
        )

    def _looks_like_video(self, value: dict[str, Any]) -> bool:
        return any(key in value for key in ("videoId", "video_id", "url", "videoUrl", "title"))

    def _runs_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            if isinstance(value.get("simpleText"), str):
                return value["simpleText"].strip()
            runs = value.get("runs")
            if isinstance(runs, list):
                return "".join(str(run.get("text", "")) for run in runs if isinstance(run, dict)).strip()
        return ""

    def _video_id_from_url(self, url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if query_id:
            return query_id
        match = re.search(r"/(?:shorts/|embed/)?([A-Za-z0-9_-]{6,})", parsed.path)
        return match.group(1) if match else ""

    def _content(self, title: str, channel: str, url: str, description: str) -> str:
        parts = [part for part in (title, f"Channel: {channel}" if channel else "", f"URL: {url}" if url else "", description) if part]
        return "\n".join(parts)
