"""Adapter for Pocket reading-list exports."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class PocketAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pocket"

    @property
    def entity_types(self) -> list[str]:
        return ["saved_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "saved_item" not in entity_types:
            return result

        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists() or not path.is_file():
            return result

        try:
            items = self._read_items(path)
        except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for item in items:
            url = self._url(item)
            title = self._title(item, url)
            if not url and not title:
                continue

            added_at = self._parse_datetime(self._first(item, "time_added", "added_at", "created_at"))
            updated_at = self._parse_datetime(
                self._first(item, "time_updated", "updated_at", "time_read", "time_favorited")
            )
            comparable_at = updated_at or added_at
            if sync_at and comparable_at and comparable_at <= sync_at:
                continue

            tags = self._tags(item.get("tags"))
            excerpt = self._first(item, "excerpt", "resolved_excerpt", "description")
            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.POCKET,
                    source_id=self._source_id(item, url, title),
                    source_entity_type="saved_item",
                    title=title or url or "Untitled Pocket item",
                    content=self._content(title, url, excerpt, tags),
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "url": url,
                        "excerpt": excerpt,
                        "status": self._first(item, "status"),
                        "time_added": self._first(item, "time_added", "added_at", "created_at"),
                        "time_updated": self._first(
                            item,
                            "time_updated",
                            "updated_at",
                            "time_read",
                            "time_favorited",
                        ),
                        "tags": tags,
                    },
                    tags=tags,
                    created_at=added_at or updated_at or datetime.now(timezone.utc),
                    updated_at=updated_at or added_at or datetime.now(timezone.utc),
                )
            )

        return result

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                return [
                    {str(key).strip(): value for key, value in row.items() if key is not None}
                    for row in csv.DictReader(handle)
                ]

        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._json_items(parsed)

    def _json_items(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []

        for key in ("list", "items", "saved_items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                return [item for item in nested.values() if isinstance(item, dict)]

        if any(isinstance(item, dict) for item in value.values()):
            return [item for item in value.values() if isinstance(item, dict)]
        return [value]

    def _title(self, item: dict[str, Any], url: str) -> str:
        return self._first(item, "resolved_title", "given_title", "title", "item_title") or url

    def _url(self, item: dict[str, Any]) -> str:
        return self._first(item, "resolved_url", "given_url", "url", "item_url")

    def _source_id(self, item: dict[str, Any], url: str, title: str) -> str:
        item_id = self._first(item, "item_id", "id", "resolved_id")
        if item_id:
            return f"pocket:{item_id}"
        if url:
            return f"url:{url}"
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()
        return f"pocket:{digest[:24]}"

    def _content(self, title: str, url: str, excerpt: str, tags: list[str]) -> str:
        parts = []
        if title:
            parts.append(title)
        if url:
            parts.append(f"URL: {url}")
        if excerpt:
            parts.append(f"Excerpt: {excerpt}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _tags(self, value: Any) -> list[str]:
        raw_tags: list[Any]
        if isinstance(value, dict):
            raw_tags = []
            for key, tag_value in value.items():
                if isinstance(tag_value, dict):
                    raw_tags.append(tag_value.get("tag") or tag_value.get("name") or key)
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
                tag = tag.get("tag") or tag.get("name") or ""
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
