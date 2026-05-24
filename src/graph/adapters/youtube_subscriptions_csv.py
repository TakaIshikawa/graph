"""Adapter for YouTube Takeout subscriptions CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class YoutubeSubscriptionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "youtube_subscriptions_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["channel_subscription"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "channel_subscription" not in entity_types:
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        channel_id = first(row, "Channel Id", "Channel ID", "ChannelId", "channel_id")
        title = first(row, "Channel Title", "Title", "Name")
        url = first(row, "Channel Url", "Channel URL", "Url", "URL")
        subscribed = parse_datetime(first(row, "Subscribed Date", "Subscription Date", "Subscribed At", "Time"))
        if not any((channel_id, title, url)):
            return None
        if not url and channel_id:
            url = f"https://www.youtube.com/channel/{channel_id}"
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "channel_id": channel_id,
                "title": title,
                "url": url,
                "source_url": url,
                "subscribed_at": subscribed.isoformat() if subscribed else "",
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="youtube_subscriptions_csv",
            source_id=f"youtube_subscriptions_csv:channel:{channel_id}" if channel_id else digest_source_id("youtube_subscriptions_csv", url or title or index),
            source_entity_type="channel_subscription",
            title=title or channel_id or url or "YouTube subscription",
            content=self._content(title, channel_id, url, subscribed),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["youtube-subscription"],
            created_at=subscribed or now,
            updated_at=subscribed or now,
        )

    def _content(self, title: str, channel_id: str, url: str, subscribed: datetime | None) -> str:
        parts = [title] if title else []
        if channel_id:
            parts.append(f"Channel ID: {channel_id}")
        if url:
            parts.append(f"URL: {url}")
        if subscribed:
            parts.append(f"Subscribed: {subscribed.isoformat()}")
        return "\n".join(parts)
