"""Adapter for Pinboard JSON bookmark exports."""

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


class PinboardAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pinboard"

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

        paths = self._json_paths()
        if not paths:
            return result

        sync_at = self._sync_datetime(since) if since else None
        for path in paths:
            for bookmark in self._read_bookmarks(path):
                unit = self._unit_from_bookmark(bookmark)
                if unit is None:
                    continue
                if sync_at and unit.created_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _json_paths(self) -> list[Path]:
        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists():
            return []
        if path.is_file():
            return [path] if path.suffix.lower() == ".json" else []
        if path.is_dir():
            return sorted(item for item in path.rglob("*.json") if item.is_file())
        return []

    def _read_bookmarks(self, path: Path) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        return self._bookmark_records(parsed)

    def _bookmark_records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            records: list[dict[str, Any]] = []
            for item in value:
                records.extend(self._bookmark_records(item))
            return records
        if not isinstance(value, dict):
            return []
        if self._has_bookmark_shape(value):
            return [value]

        records = []
        for key in ("posts", "bookmarks", "items", "results", "list"):
            nested = value.get(key)
            if isinstance(nested, (dict, list)):
                records.extend(self._bookmark_records(nested))
        if records:
            return records

        for item in value.values():
            records.extend(self._bookmark_records(item))
        return records

    def _unit_from_bookmark(self, bookmark: dict[str, Any]) -> KnowledgeUnit | None:
        url = self._first(bookmark, "href", "url")
        title = self._first(bookmark, "description", "title") or url
        notes = self._first(bookmark, "extended", "notes", "note")
        if not url and not title:
            return None

        time_text = self._first(bookmark, "time", "created_at", "created")
        created_at = self._parse_datetime(time_text)
        tags = self._tags(bookmark.get("tags") or bookmark.get("tag"))
        shared = self._first(bookmark, "shared")
        toread = self._first(bookmark, "toread")
        item_hash = self._first(bookmark, "hash")

        return KnowledgeUnit(
            source_project=SourceProject.PINBOARD,
            source_id=self._source_id(item_hash, url, title, notes),
            source_entity_type="bookmark",
            title=title or "Untitled Pinboard bookmark",
            content=self._content(title, url, notes, tags),
            content_type=ContentType.ARTIFACT,
            metadata={
                "url": url,
                "notes": notes,
                "hash": item_hash,
                "shared": shared,
                "toread": toread,
                "time": time_text,
                "tags": tags,
            },
            tags=tags,
            created_at=created_at or datetime.now(timezone.utc),
            updated_at=created_at or datetime.now(timezone.utc),
        )

    def _has_bookmark_shape(self, value: dict[str, Any]) -> bool:
        return any(key in value for key in ("href", "url", "description", "title")) and any(
            key in value for key in ("href", "url", "hash")
        )

    def _source_id(self, item_hash: str, url: str, title: str, notes: str) -> str:
        if item_hash:
            return f"pinboard:{item_hash}"
        if url:
            return f"url:{url}"
        digest = hashlib.sha256(f"{title}\n{notes}".encode("utf-8")).hexdigest()
        return f"pinboard:{digest[:24]}"

    def _content(self, title: str, url: str, notes: str, tags: list[str]) -> str:
        parts = []
        if title:
            parts.append(title)
        if url:
            parts.append(f"URL: {url}")
        if notes:
            parts.append(f"Notes: {notes}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _tags(self, value: Any) -> list[str]:
        if isinstance(value, dict):
            raw_tags = value.values()
        elif isinstance(value, list):
            raw_tags = value
        elif isinstance(value, str):
            raw_tags = re.split(r"[\s,;|]+", value)
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
