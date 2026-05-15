"""Adapter for generic RSS reader starred-item JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class RssReaderStarredJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "rss_reader_starred_json"

    @property
    def entity_types(self) -> list[str]:
        return ["starred_feed_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "starred_feed_item" not in set(entity_types or self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                items = self._read_items(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for item in items:
                unit = self._unit_from_item(item, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.json") if child.is_file())

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._items(parsed)

    def _items(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []
        for key in ("items", "entries", "starred", "saved", "data", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                items = self._items(nested)
                if items:
                    return items
        return [value]

    def _unit_from_item(self, item: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = self._first(item, "title", "name")
        url = self._first(item, "url", "link", "canonical_url", "href")
        feed_title = self._feed_title(item)
        author = self._first(item, "author", "byline", "creator")
        summary = self._first(item, "summary", "excerpt", "description", "content_text", "content")
        tags = self._tags(item.get("tags") or item.get("categories") or item.get("labels"))
        published_at = self._parse_datetime(self._first(item, "published_at", "published", "date_published", "created_at"))
        starred_at = self._parse_datetime(self._first(item, "starred_at", "starred", "saved_at", "marked_at", "updated_at"))
        if not title and not url and not summary:
            return None
        metadata = {
            "title": title,
            "url": url,
            "feed_title": feed_title,
            "author": author,
            "summary": summary,
            "tags": tags,
            "published_at": published_at.isoformat() if published_at else self._first(item, "published_at", "published", "date_published", "created_at"),
            "starred_at": starred_at.isoformat() if starred_at else self._first(item, "starred_at", "starred", "saved_at", "marked_at", "updated_at"),
            "source_file": source_file,
            "item": item,
        }
        now = datetime.now(timezone.utc)
        created = published_at or starred_at or now
        updated = starred_at or published_at or now
        return KnowledgeUnit(
            source_project=SourceProject.RSS_READER_STARRED_JSON,
            source_id=self._source_id(url, title, feed_title, published_at),
            source_entity_type="starred_feed_item",
            title=title or url,
            content=self._content(metadata),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(["rss", "starred_feed_item", *tags])),
            created_at=created,
            updated_at=updated,
        )

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [metadata.get("title"), metadata.get("summary")]
        for key, label in (("feed_title", "Feed"), ("author", "Author"), ("url", "URL")):
            if metadata.get(key):
                parts.append(f"{label}: {metadata[key]}")
        return "\n\n".join(str(item) for item in parts if item)

    def _source_id(self, url: str, title: str, feed_title: str, published_at: datetime | None) -> str:
        raw = url or "|".join([title.casefold().strip(), feed_title.casefold().strip(), published_at.isoformat() if published_at else ""])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"rss_reader_starred_json:{digest}"

    def _feed_title(self, item: dict[str, Any]) -> str:
        feed = item.get("feed")
        if isinstance(feed, dict):
            return self._text(feed.get("title") or feed.get("name"))
        return self._first(item, "feed_title", "feed", "source", "source_title")

    def _tags(self, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            raw = value.replace(";", ",").replace("|", ",").split(",")
        else:
            raw = [item.get("name") if isinstance(item, dict) else item for item in value] if isinstance(value, list) else []
        tags: list[str] = []
        for item in raw:
            tag = self._text(item)
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is not None and not isinstance(value, (dict, list)) and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_datetime(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
