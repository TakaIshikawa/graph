"""Adapter for Wallabag read-it-later article exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class WallabagAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "wallabag"

    @property
    def entity_types(self) -> list[str]:
        return ["article"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "article" not in entity_types:
            return result

        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists() or not path.is_file():
            return result

        try:
            items = self._read_items(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for item in items:
            url = self._url(item)
            title = self._title(item, url)
            if not url and not title:
                continue

            created_text = self._first(item, "created_at", "created", "date_created")
            updated_text = self._first(item, "updated_at", "updated", "date_updated")
            created_at = self._parse_datetime(created_text)
            updated_at = self._parse_datetime(updated_text)
            comparable_at = updated_at or created_at
            if sync_at and comparable_at and comparable_at <= sync_at:
                continue

            content = self._first(item, "content", "body", "html")
            tags = self._parse_tags(item.get("tags") or item.get("tag"))
            reading_time = self._first(item, "reading_time", "readingTime")
            archived = self._first(item, "is_archived", "archived")
            starred = self._first(item, "is_starred", "starred", "favorite")
            wallabag_id = self._first(item, "id", "item_id", "entry_id")

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.WALLABAG,
                    source_id=self._source_id(wallabag_id, url, title),
                    source_entity_type="article",
                    title=title or url or "Untitled Wallabag article",
                    content=self._content(title, url, content),
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "url": url,
                        "content": content,
                        "reading_time": reading_time,
                        "archived": archived,
                        "starred": starred,
                        "tags": tags,
                        "created_at": created_text,
                        "updated_at": updated_text,
                        "wallabag_id": wallabag_id,
                    },
                    tags=tags,
                    created_at=created_at or updated_at or datetime.now(timezone.utc),
                    updated_at=updated_at or created_at or datetime.now(timezone.utc),
                )
            )

        return result

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._json_items(parsed)

    def _json_items(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []

        for key in ("entries", "items", "articles", "list"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                return [item for item in nested.values() if isinstance(item, dict)]

        if any(isinstance(item, dict) for item in value.values()):
            return [item for item in value.values() if isinstance(item, dict)]
        return [value]

    def _title(self, item: dict[str, Any], url: str) -> str:
        return self._first(item, "title", "entry_title", "name") or url

    def _url(self, item: dict[str, Any]) -> str:
        return self._first(item, "url", "href", "link")

    def _source_id(self, wallabag_id: str, url: str, title: str) -> str:
        if wallabag_id:
            return f"wallabag:{wallabag_id}"
        if url:
            return f"url:{url}"
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()
        return f"wallabag:{digest[:24]}"

    def _content(self, title: str, url: str, content: str) -> str:
        parts = []
        if title:
            parts.append(title)
        if url:
            parts.append(f"URL: {url}")
        if content:
            # Truncate very long content
            if len(content) > 5000:
                content = content[:5000] + "..."
            parts.append(f"Content: {content}")
        return "\n".join(parts)

    def _parse_tags(self, value: Any) -> list[str]:
        raw_tags: list[Any]
        if isinstance(value, dict):
            raw_tags = []
            for key, tag_value in value.items():
                if isinstance(tag_value, dict):
                    raw_tags.append(tag_value.get("label") or tag_value.get("tag") or tag_value.get("name") or key)
                else:
                    raw_tags.append(tag_value or key)
        elif isinstance(value, list):
            raw_tags = value
        elif isinstance(value, str):
            raw_tags = re.split(r"[,;|]", value)
        else:
            raw_tags = []

        tags: list[str] = []
        for tag in raw_tags:
            if isinstance(tag, dict):
                tag = tag.get("label") or tag.get("tag") or tag.get("name") or ""
            normalized = re.sub(r"\s+", " ", str(tag).strip().removeprefix("#")).strip().lower()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        if re.fullmatch(r"\d+(?:\.0+)?", value):
            try:
                return datetime.fromtimestamp(int(float(value)), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
