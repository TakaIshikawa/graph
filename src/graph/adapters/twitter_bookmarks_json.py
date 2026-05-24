"""Adapter for Twitter/X archive bookmarks JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class TwitterBookmarksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "twitter_bookmarks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["tweet"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "tweet" not in entity_types:
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        for path in iter_paths(self.path, {".json", ".js"}):
            for index, record in enumerate(self._records(path)):
                unit = self._unit_from_record(record, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _records(self, path: Path) -> list[dict[str, Any]]:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        if "=" in text[:200] and not text.lstrip().startswith(("[", "{")):
            text = text.split("=", 1)[1].strip().rstrip(";")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("bookmarks") or data.get("tweets") or data.get("data") or []
        else:
            items = []
        return [item for item in (self._unwrap(item) for item in items) if isinstance(item, dict)]

    def _unwrap(self, value: Any) -> Any:
        while isinstance(value, dict):
            for key in ("tweet", "tweet_results", "result", "bookmark"):
                nested = value.get(key)
                if isinstance(nested, dict):
                    value = nested
                    break
            else:
                return value
        return value

    def _unit_from_record(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        legacy = record.get("legacy") if isinstance(record.get("legacy"), dict) else {}
        tweet_id = self._first((record, legacy), ("id_str", "id", "tweet_id", "tweetId", "rest_id"))
        text = self._first((record, legacy), ("full_text", "text", "content"))
        author = self._author(record)
        created = parse_datetime(self._first((record, legacy), ("created_at", "createdAt")))
        bookmarked = parse_datetime(self._first((record, legacy), ("bookmarked_at", "bookmarkedAt", "saved_at")))
        if not any((tweet_id, text, author)):
            return None
        url = self._first((record, legacy), ("url", "tweet_url", "tweetUrl")) or (f"https://twitter.com/{author}/status/{tweet_id}" if author and tweet_id else "")
        media_urls = self._media_urls(record, legacy)
        conversation_id = self._first((record, legacy), ("conversation_id_str", "conversation_id", "conversationId"))
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "tweet_id": tweet_id,
                "author_handle": author,
                "text": text,
                "url": url,
                "source_url": url,
                "created_at": created.isoformat() if created else "",
                "bookmarked_at": bookmarked.isoformat() if bookmarked else "",
                "media_urls": media_urls,
                "conversation_id": conversation_id,
                "source_file": source_file,
                "row": record,
            }
        )
        return KnowledgeUnit(
            source_project="twitter_bookmarks_json",
            source_id=f"twitter_bookmarks_json:tweet:{tweet_id}" if tweet_id else digest_source_id("twitter_bookmarks_json", author, text, index),
            source_entity_type="tweet",
            title=(text[:80] if text else f"Tweet {tweet_id}"),
            content=self._content(text, author, url, media_urls, conversation_id),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["twitter-bookmark"],
            created_at=created or bookmarked or now,
            updated_at=bookmarked or created or now,
        )

    def _first(self, dicts: tuple[dict[str, Any], ...], keys: tuple[str, ...]) -> str:
        for key in keys:
            for item in dicts:
                value = item.get(key) if isinstance(item, dict) else None
                if value is not None and str(value).strip():
                    return str(value).strip()
        return ""

    def _author(self, record: dict[str, Any]) -> str:
        for value in (record.get("author_handle"), record.get("screen_name"), record.get("username")):
            if value:
                return str(value).lstrip("@")
        core = record.get("core")
        user = core.get("user_results", {}).get("result", {}) if isinstance(core, dict) else {}
        legacy_user = user.get("legacy", {}) if isinstance(user, dict) else {}
        return str(legacy_user.get("screen_name") or "").lstrip("@")

    def _media_urls(self, record: dict[str, Any], legacy: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        media = record.get("media") or record.get("entities", {}).get("media", []) or legacy.get("media") or legacy.get("entities", {}).get("media", [])
        if isinstance(media, dict):
            media = media.values()
        for item in media if isinstance(media, list) else []:
            if isinstance(item, dict):
                url = item.get("media_url_https") or item.get("media_url") or item.get("url")
                if url and str(url) not in urls:
                    urls.append(str(url))
        return urls

    def _content(self, text: str, author: str, url: str, media_urls: list[str], conversation_id: str) -> str:
        parts = [text] if text else []
        if author:
            parts.append(f"Author: @{author}")
        if url:
            parts.append(f"URL: {url}")
        if conversation_id:
            parts.append(f"Conversation: {conversation_id}")
        if media_urls:
            parts.append(f"Media: {', '.join(media_urls)}")
        return "\n".join(parts)
