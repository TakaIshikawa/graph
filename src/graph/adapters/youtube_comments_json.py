"""Adapter for YouTube comment JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class YoutubeCommentsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "youtube_comments_json"

    @property
    def entity_types(self) -> list[str]:
        return ["comment"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "comment" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            for index, record in enumerate(_records(path, ("comments", "items", "data"))):
                unit = self._unit(record, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        comment_id = first(record, "id", "comment_id", "commentId")
        text = first(record, "text", "comment", "content", "body")
        video_id = first(record, "video_id", "videoId")
        video_title = first(record, "video_title", "videoTitle", "title")
        if not any([comment_id, text, video_id, video_title]):
            return None
        published = parse_datetime(first(record, "published_at", "publishedAt", "created_at", "createdAt"))
        updated = parse_datetime(first(record, "updated_at", "updatedAt")) or published or datetime.now(timezone.utc)
        metadata = clean_metadata({"comment_id": comment_id, "text": text, "video_id": video_id, "video_title": video_title, "channel_name": first(record, "channel_name", "channelName", "channel"), "url": first(record, "url", "video_url", "videoUrl"), "parent_id": first(record, "parent_id", "parentId"), "like_count": parse_int(first(record, "like_count", "likeCount")), "published_at": published.isoformat() if published else first(record, "published_at", "publishedAt"), "updated_at": updated.isoformat(), "source_file": source_file})
        title = video_title or f"YouTube comment {comment_id or index + 1}"
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{comment_id}" if comment_id else digest_source_id(self.name, video_id, text, published, index), source_entity_type="comment", title=title, content=_content(text, video_title, metadata.get("url", "")), content_type=ContentType.ARTIFACT, metadata=metadata, tags=["youtube", "comment"], created_at=published or updated, updated_at=updated)


def _records(path: Path, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    raw = parsed
    if isinstance(parsed, dict):
        for key in keys:
            if isinstance(parsed.get(key), list):
                raw = parsed[key]
                break
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])


def _content(text: str, title: str, url: str) -> str:
    return "\n".join(part for part in (text, f"Video: {title}" if title else "", f"URL: {url}" if url else "") if part)
