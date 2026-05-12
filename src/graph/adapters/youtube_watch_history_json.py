"""Adapter for Google Takeout YouTube watch history JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class YouTubeWatchHistoryJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "youtube_watch_history_json"

    @property
    def entity_types(self) -> list[str]:
        return ["watch"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or self.entity_types)
        if "watch" not in allowed:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit_from_record(record, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, dict):
            for key in ("watchHistory", "watch_history", "history", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    parsed = value
                    break
        return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []

    def _unit_from_record(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        watched_at = parse_datetime(record.get("time"))
        title = self._text(record.get("title"))
        url = self._text(record.get("titleUrl") or record.get("url"))
        if watched_at is None or not (title or url):
            return None

        channel_name, channel_url = self._channel(record)
        products = self._text_list(record.get("products"))
        subtitles = record.get("subtitles") if isinstance(record.get("subtitles"), list) else []
        metadata = clean_metadata(
            {
                "title": title,
                "title_url": url,
                "watched_at": watched_at.isoformat(),
                "channel_name": channel_name,
                "channel_url": channel_url,
                "products": products,
                "subtitles": subtitles,
                "source_file": source_file,
                "record": dict(record),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.YOUTUBE_WATCH_HISTORY_JSON,
            source_id=self._source_id(url, title, watched_at, index),
            source_entity_type="watch",
            title=title or "YouTube watch",
            content=self._content(title, url, watched_at, channel_name, channel_url, products),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["youtube", "watch"],
            created_at=watched_at,
            updated_at=watched_at,
        )

    def _source_id(self, url: str, title: str, watched_at: datetime, index: int) -> str:
        identity = url or title
        if not identity:
            identity = str(index)
        return digest_source_id("youtube_watch_history_json", identity, watched_at.isoformat())

    def _channel(self, record: dict[str, Any]) -> tuple[str, str]:
        subtitles = record.get("subtitles")
        if isinstance(subtitles, list):
            for item in subtitles:
                if not isinstance(item, dict):
                    continue
                name = self._text(item.get("name"))
                url = self._text(item.get("url"))
                if name or url:
                    return name, url
        details = record.get("details")
        if isinstance(details, list):
            for item in details:
                if isinstance(item, dict):
                    name = self._text(item.get("name"))
                    url = self._text(item.get("url"))
                    if name or url:
                        return name, url
        return "", ""

    def _content(self, title: str, url: str, watched_at: datetime, channel_name: str, channel_url: str, products: list[str]) -> str:
        parts = [f"Watched: {title or url}", f"Watched at: {watched_at.isoformat()}"]
        if url:
            parts.append(f"URL: {url}")
        if channel_name:
            parts.append(f"Channel: {channel_name}")
        if channel_url:
            parts.append(f"Channel URL: {channel_url}")
        if products:
            parts.append(f"Products: {', '.join(products)}")
        return "\n".join(parts)

    def _text_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [text for item in value if (text := self._text(item))]

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
