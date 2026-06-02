"""Adapter for Slack bookmark JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SlackBookmarksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "slack_bookmarks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "bookmark" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            for index, record in enumerate(self._records(path)):
                unit = self._unit(record, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _records(self, path: Path) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        return list(_walk_records(parsed))

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        bookmark_id = first(record, "id", "bookmark_id", "bookmarkId")
        title = first(record, "title", "name", "text")
        link = first(record, "link", "url", "entity_url", "entityUrl")
        channel = first(record, "channel_name", "channelName", "channel", "channel_id", "channelId")
        if not any([bookmark_id, title, link]):
            return None
        channel_id = first(record, "channel_id", "channelId")
        user = first(record, "user_id", "userId", "user", "created_by", "createdBy")
        entity_type = first(record, "entity_type", "entityType", "type") or "link"
        created = parse_datetime(first(record, "date_created", "dateCreated", "created_at", "createdAt"))
        updated = parse_datetime(first(record, "date_updated", "dateUpdated", "updated_at", "updatedAt")) or created
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "bookmark_id": bookmark_id,
                "title": title,
                "link": link,
                "url": link,
                "channel": channel,
                "channel_id": channel_id,
                "user": user,
                "date_created": created.isoformat() if created else first(record, "date_created", "dateCreated", "created_at", "createdAt"),
                "date_updated": updated.isoformat() if updated else first(record, "date_updated", "dateUpdated", "updated_at", "updatedAt"),
                "entity_type": entity_type,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=self.name,
            source_id=f"{self.name}:{bookmark_id}" if bookmark_id else digest_source_id(self.name, title, link, channel, index),
            source_entity_type="bookmark",
            title=title or link or f"Slack bookmark {index + 1}",
            content=_content(title, link, channel),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=[tag for tag in dict.fromkeys(["slack", "bookmark", channel]) if tag],
            created_at=created or updated or now,
            updated_at=updated or created or now,
        )


def _walk_records(value: Any, channel: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            records.extend(_walk_records(item, channel))
    elif isinstance(value, dict):
        next_channel = channel
        if isinstance(value.get("bookmarks"), list) and any(key in value for key in ("id", "name", "channel_id", "channel_name")):
            next_channel = value
        if "bookmarks" in value:
            for item in _walk_records(value["bookmarks"], next_channel):
                records.append(item)
        for key in ("items", "data"):
            if key in value:
                records.extend(_walk_records(value[key], next_channel))
        if "channels" in value:
            records.extend(_walk_records(value["channels"], next_channel))
        if any(key in value for key in ("link", "url", "entity_url", "title")) and "bookmarks" not in value:
            record = dict(value)
            if channel:
                record.setdefault("channel_id", first(channel, "id", "channel_id", "channelId"))
                record.setdefault("channel_name", first(channel, "name", "channel_name", "channelName"))
            records.append(record)
    return records


def _content(title: str, link: str, channel: str) -> str:
    parts = [title, f"URL: {link}" if link else "", f"Channel: {channel}" if channel else ""]
    return "\n".join(part for part in parts if part)
