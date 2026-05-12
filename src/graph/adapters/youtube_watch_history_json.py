"""Adapter for Google Takeout YouTube watch history JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class YouTubeWatchHistoryJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "youtube_watch_history_json"

    @property
    def entity_types(self) -> list[str]:
        return ["channel", "watch"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else {"watch"}
        if not allowed.intersection(self.entity_types):
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
                if unit.source_entity_type == "watch":
                    result.units.append(unit)

        watches = sorted(result.units, key=lambda unit: (unit.updated_at, unit.source_id))
        channels = self._channel_units(watches) if "channel" in allowed else []
        result.units = []
        if "channel" in allowed:
            result.units.extend(channels)
        if "watch" in allowed:
            result.units.extend(watches)
        if {"channel", "watch"}.issubset(allowed):
            result.edges.extend(self._channel_watch_edges(channels, watches))
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
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

    def _channel_units(self, watches: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[tuple[str, str], list[KnowledgeUnit]] = {}
        for watch in watches:
            key = self._channel_identity(watch.metadata)
            if key[1]:
                grouped.setdefault(key, []).append(watch)

        units: list[KnowledgeUnit] = []
        for identity, channel_watches in grouped.items():
            first = channel_watches[0]
            name = str(first.metadata.get("channel_name") or "")
            url = str(first.metadata.get("channel_url") or "")
            created_at = min(watch.created_at for watch in channel_watches)
            updated_at = max(watch.updated_at for watch in channel_watches)
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.YOUTUBE_WATCH_HISTORY_JSON,
                    source_id=self._channel_source_id(identity),
                    source_entity_type="channel",
                    title=name or url or "YouTube channel",
                    content=f"YouTube channel: {name or url}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "channel": name or url,
                        "channel_name": name,
                        "channel_url": url,
                        "watch_count": len(channel_watches),
                        "video_source_ids": [watch.source_id for watch in channel_watches],
                        "first_watched_at": created_at.isoformat(),
                        "latest_watched_at": updated_at.isoformat(),
                        "last_watched_at": updated_at.isoformat(),
                        "source_files": sorted({str(watch.metadata.get("source_file")) for watch in channel_watches if watch.metadata.get("source_file")}),
                        "watched_video_source_ids": [watch.source_id for watch in channel_watches],
                    },
                    tags=["youtube", "channel"],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _channel_watch_edges(self, channels: list[KnowledgeUnit], watches: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        channel_ids = {self._channel_identity(channel.metadata): channel.source_id for channel in channels}
        edges: list[KnowledgeEdge] = []
        for watch in watches:
            channel_id = channel_ids.get(self._channel_identity(watch.metadata))
            if channel_id:
                edges.append(self._edge(channel_id, watch.source_id, "channel_contains_watch"))
        return list({edge.id: edge for edge in edges}.values())

    def _channel_identity(self, metadata: dict[str, Any]) -> tuple[str, str]:
        url = self._text(metadata.get("channel_url")).casefold()
        if url:
            return ("url", url)
        return ("name", " ".join(self._text(metadata.get("channel_name")).casefold().split()))

    def _channel_source_id(self, identity: tuple[str, str]) -> str:
        return digest_source_id("youtube_watch_history_json:channel", *identity)

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=digest_source_id("youtube-watch-history-json-edge", from_id, to_id, relation_type),
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.YOUTUBE_WATCH_HISTORY_JSON.value,
                "relation_type": relation_type,
            },
        )

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
