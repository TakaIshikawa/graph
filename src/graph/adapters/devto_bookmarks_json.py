"""Adapter for DEV.to bookmarked article JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class DevtoBookmarksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "devto_bookmarks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["article_bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else set(self.entity_types)
        if "article_bookmark" not in allowed:
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for record in records:
                unit = self._unit_from_record(record, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.updated_at, unit.source_id)))
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        return self._records(json.loads(path.read_text(encoding="utf-8-sig")))

    def _records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []
        for key in ("articles", "items", "bookmarks", "saved", "data", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                records = self._records(nested)
                if records:
                    return records
        return [value] if self._looks_like_record(value) else []

    def _looks_like_record(self, value: dict[str, Any]) -> bool:
        return bool(first(value, "title", "url", "canonical_url", "path", "id"))

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = first(record, "title")
        url = self._article_url(record)
        if not title and not url:
            return None

        published_at = self._record_datetime(record, "published_at", "publishedAt", "published_timestamp")
        saved_at = self._record_datetime(record, "saved_at", "savedAt", "bookmarked_at", "bookmarkedAt", "created_at", "createdAt")
        updated_at = saved_at or self._record_datetime(record, "updated_at", "updatedAt") or published_at
        now = datetime.now(timezone.utc)
        event_at = updated_at or published_at or now
        tags = split_values(record.get("tags") or record.get("tag_list") or record.get("tagList"))
        author_username = self._author_username(record)
        metadata = clean_metadata(
            {
                "article_id": parse_int(first(record, "id", "article_id", "articleId")),
                "title": title,
                "url": url,
                "author_username": author_username,
                "tags": tags,
                "published_at": published_at.isoformat() if published_at else None,
                "saved_at": saved_at.isoformat() if saved_at else None,
                "positive_reactions_count": parse_int(record.get("positive_reactions_count") or record.get("positiveReactionsCount")),
                "reading_time_minutes": parse_int(record.get("reading_time_minutes") or record.get("readingTimeMinutes")),
                "source_file": source_file,
                "record": dict(record),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.DEVTO_BOOKMARKS_JSON,
            source_id=digest_source_id("devto_bookmarks_json", url or first(record, "id", "article_id", "articleId") or title),
            source_entity_type="article_bookmark",
            title=title or url,
            content=self._content(title, url, author_username, tags, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["devto", "article_bookmark", *tags],
            created_at=published_at or event_at,
            updated_at=event_at,
        )

    def _article_url(self, record: dict[str, Any]) -> str:
        canonical = first(record, "canonical_url", "canonicalUrl")
        if canonical:
            return canonical
        url = first(record, "url", "article_url", "articleUrl")
        if url:
            return url
        path = first(record, "path")
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"https://dev.to{path}" if path.startswith("/") else path

    def _author_username(self, record: dict[str, Any]) -> str:
        user = record.get("user") or record.get("author")
        if isinstance(user, dict):
            return first(user, "username", "name")
        return first(record, "user_username", "username", "author_username", "authorUsername")

    def _content(self, title: str, url: str, author_username: str, tags: list[str], metadata: dict[str, Any]) -> str:
        parts = [title] if title else []
        if url:
            parts.append(f"URL: {url}")
        if author_username:
            parts.append(f"Author: {author_username}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        for label, key in (("Positive reactions", "positive_reactions_count"), ("Reading time minutes", "reading_time_minutes")):
            if key in metadata:
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(parts)

    def _record_datetime(self, record: dict[str, Any], *keys: str) -> datetime | None:
        for key in keys:
            parsed = self._parse_datetime_value(record.get(key))
            if parsed is not None:
                return parsed
        return None

    def _parse_datetime_value(self, value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        text = str(value).strip()
        if text.isdigit():
            try:
                return datetime.fromtimestamp(float(text), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        return parse_datetime(text)
