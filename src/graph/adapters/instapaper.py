"""Adapter for Instapaper reading-list exports."""

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


class InstapaperAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "instapaper"

    @property
    def entity_types(self) -> list[str]:
        return ["bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "bookmark" not in entity_types:
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

            created_text = self._first(item, "time", "timestamp", "created_at", "created")
            created_at = self._parse_datetime(created_text)
            if sync_at and created_at and created_at <= sync_at:
                continue

            description = self._first(item, "description", "summary", "excerpt")
            progress = self._first(item, "progress", "reading_progress")
            starred = self._first(item, "starred")
            folder = self._first(item, "folder", "folder_id", "folder_name")
            bookmark_hash = self._first(item, "hash", "bookmark_id", "id")

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.INSTAPAPER,
                    source_id=self._source_id(item, url, title, bookmark_hash),
                    source_entity_type="bookmark",
                    title=title or url or "Untitled Instapaper bookmark",
                    content=self._content(title, url, description, folder),
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "url": url,
                        "description": description,
                        "progress": progress,
                        "starred": starred,
                        "folder": folder,
                        "hash": bookmark_hash,
                        "time": created_text,
                    },
                    tags=[],
                    created_at=created_at or datetime.now(timezone.utc),
                    updated_at=created_at or datetime.now(timezone.utc),
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

        for key in ("bookmarks", "items", "articles"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                return [item for item in nested.values() if isinstance(item, dict)]

        if any(isinstance(item, dict) for item in value.values()):
            return [item for item in value.values() if isinstance(item, dict)]
        return [value]

    def _title(self, item: dict[str, Any], url: str) -> str:
        return self._first(item, "title", "bookmark_title") or url

    def _url(self, item: dict[str, Any]) -> str:
        return self._first(item, "url", "href")

    def _source_id(self, item: dict[str, Any], url: str, title: str, bookmark_hash: str) -> str:
        bookmark_id = self._first(item, "bookmark_id", "id")
        if bookmark_id:
            return f"instapaper:{bookmark_id}"
        if bookmark_hash:
            return f"instapaper:{bookmark_hash}"
        if url:
            return f"url:{url}"
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()
        return f"instapaper:{digest[:24]}"

    def _content(self, title: str, url: str, description: str, folder: str) -> str:
        parts = []
        if title:
            parts.append(title)
        if url:
            parts.append(f"URL: {url}")
        if description:
            parts.append(f"Description: {description}")
        if folder:
            parts.append(f"Folder: {folder}")
        return "\n".join(parts)

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
