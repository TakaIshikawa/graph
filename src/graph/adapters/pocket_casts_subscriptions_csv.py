"""Adapter for Pocket Casts podcast subscriptions CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import ensure_utc, first, iter_paths, parse_datetime, parse_int, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PocketCastsSubscriptionsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pocket_casts_subscriptions_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["podcast_subscription"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "podcast_subscription" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows, start=1):
                unit = self._unit_from_row(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, source_row: int) -> KnowledgeUnit | None:
        title = first(row, "Podcast Title", "podcast_title", "Podcast", "Title", "Show", "Show Title")
        author = first(row, "Author", "Publisher", "Podcast Author")
        feed_url = first(row, "Feed URL", "feed_url", "RSS URL", "Rss Feed", "Podcast URL")
        website_url = first(row, "Website URL", "website_url", "Website", "URL", "Link")
        description = first(row, "Description", "Summary")
        categories = split_values(first(row, "Categories", "Category", "Genres", "Tags"))
        episode_count = parse_int(first(row, "Episode Count", "Episodes", "Total Episodes", "episode_count"))
        subscribed_at_text = first(row, "Subscribed At", "Subscription Date", "Subscribed", "Date Subscribed")
        last_published_text = first(row, "Last Published At", "Last Published", "Latest Episode At", "Updated At")
        subscribed_at = parse_datetime(subscribed_at_text)
        last_published_at = parse_datetime(last_published_text)
        if not title and not feed_url:
            return None

        metadata = {
            "title": title,
            "author": author,
            "feed_url": feed_url,
            "website_url": website_url,
            "description": description,
            "categories": categories,
            "episode_count": episode_count,
            "subscribed_at": subscribed_at.isoformat() if subscribed_at else subscribed_at_text,
            "last_published_at": last_published_at.isoformat() if last_published_at else last_published_text,
            "source_file": source_file,
            "source_row": source_row,
            "row": dict(row),
        }
        metadata = {key: value for key, value in metadata.items() if value not in ("", None, [])}
        now = datetime.now(timezone.utc)
        updated_at = last_published_at or subscribed_at
        return KnowledgeUnit(
            source_project="pocket_casts_subscriptions_csv",
            source_id=self._source_id(feed_url, title),
            source_entity_type="podcast_subscription",
            title=title or feed_url,
            content=self._content(metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(item for item in ["pocket_casts", "podcast", "subscription", author] if item)),
            created_at=subscribed_at or updated_at or now,
            updated_at=updated_at or now,
        )

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [str(metadata.get("title") or metadata.get("feed_url") or "Pocket Casts podcast")]
        for key, label in (
            ("author", "Author"),
            ("feed_url", "Feed URL"),
            ("website_url", "Website URL"),
            ("categories", "Categories"),
            ("episode_count", "Episode count"),
            ("subscribed_at", "Subscribed"),
            ("last_published_at", "Last published"),
        ):
            if key in metadata:
                value = ", ".join(metadata[key]) if isinstance(metadata[key], list) else metadata[key]
                parts.append(f"{label}: {value}")
        if metadata.get("description"):
            parts.append(f"\n{metadata['description']}")
        return "\n".join(parts)

    def _source_id(self, feed_url: str, title: str) -> str:
        source = feed_url.strip() or self._normalized(title)
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
        return f"pocket_casts_subscriptions_csv:{digest}"

    def _normalized(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
