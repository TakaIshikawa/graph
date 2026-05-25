"""Adapter for Pinboard bookmark JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, parse_datetime, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PinboardBookmarksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pinboard_bookmarks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "bookmark" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        if sync_at and sync_at.tzinfo is None:
            sync_at = sync_at.replace(tzinfo=timezone.utc)

        for path in self._iter_paths():
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(self._records(data)):
                unit = self._unit_from_record(record, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if root.is_dir():
            return sorted(path for path in root.rglob("*.json") if path.is_file())
        return []

    def _records(self, data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("bookmarks", "posts", "items"):
                items = data.get(key)
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
            return [data]
        return []

    def _unit_from_record(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        url = self._text(record.get("href") or record.get("url"))
        title = self._text(record.get("description") or record.get("title"))
        notes = self._text(record.get("extended") or record.get("note"))
        if not any((url, title, notes)):
            return None

        timestamp_text = self._text(record.get("time") or record.get("created_at") or record.get("created"))
        timestamp = parse_datetime(timestamp_text)
        now = datetime.now(timezone.utc)
        tags = self._tags(record.get("tags") or record.get("tag"))
        shared = self._bool(record.get("shared"))
        toread = self._bool(record.get("toread") or record.get("to_read"))
        hash_value = self._text(record.get("hash"))
        meta = self._text(record.get("meta"))

        metadata = clean_metadata(
            {
                "url": url,
                "source_url": url,
                "external_url": url,
                "description": title,
                "extended": notes,
                "meta": meta,
                "hash": hash_value,
                "time": timestamp_text,
                "created_at": timestamp.isoformat() if timestamp else None,
                "shared": shared,
                "toread": toread,
                "tags": tags,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="pinboard_bookmarks_json",
            source_id=self._source_id(url, hash_value, title, index),
            source_entity_type="bookmark",
            title=title or url or "Untitled Pinboard bookmark",
            content=self._content(title, url, notes, tags, shared, toread),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=timestamp or now,
            updated_at=timestamp or now,
        )

    def _source_id(self, url: str, hash_value: str, title: str, index: int) -> str:
        return digest_source_id("pinboard_bookmarks_json", url or hash_value or title or index)

    def _content(self, title: str, url: str, notes: str, tags: list[str], shared: bool, toread: bool) -> str:
        parts = [part for part in (title, f"URL: {url}" if url else "") if part]
        if notes:
            parts.append(f"Notes: {notes}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        parts.append(f"Shared: {str(shared).lower()}")
        parts.append(f"To read: {str(toread).lower()}")
        return "\n".join(parts)

    def _tags(self, value: Any) -> list[str]:
        tags: list[str] = []
        raw = value.split() if isinstance(value, str) else split_values(value)
        for tag in raw:
            normalized = " ".join(str(tag).removeprefix("#").casefold().split())
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().casefold() in {"1", "true", "yes", "y"}

    def _text(self, value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""
        return str(value).strip()
