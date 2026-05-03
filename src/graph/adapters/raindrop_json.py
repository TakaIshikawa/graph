"""Adapter for Raindrop.io JSON bookmark exports."""

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


class RaindropJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "raindrop_json"

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
            items = self._json_items(json.loads(path.read_text(encoding="utf-8-sig")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for item in items:
            unit = self._unit_from_item(item, path.name)
            if unit is None:
                continue
            comparable_at = unit.updated_at or unit.created_at
            if sync_at and comparable_at <= sync_at:
                continue
            result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _json_items(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            records: list[dict[str, Any]] = []
            for item in value:
                records.extend(self._json_items(item))
            return records
        if not isinstance(value, dict):
            return []
        if self._has_bookmark_shape(value):
            return [value]

        records = []
        for key in ("items", "bookmarks", "raindrops", "results", "list"):
            nested = value.get(key)
            if isinstance(nested, (dict, list)):
                records.extend(self._json_items(nested))
        if records:
            return records

        for item in value.values():
            records.extend(self._json_items(item))
        return records

    def _unit_from_item(self, item: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        url = self._first(item, "link", "url", "href")
        if not url:
            return None

        title = self._first(item, "title", "name") or url
        description = self._first(item, "excerpt", "description", "summary")
        notes = self._notes(item.get("note") or item.get("notes"))
        tags = self._tags(item.get("tags") or item.get("tag"))
        collection = self._collection(item.get("collection") or item.get("folder"))
        created_text = self._first(item, "created", "created_at", "createdAt", "date")
        updated_text = self._first(
            item,
            "lastUpdate",
            "last_update",
            "updated",
            "updated_at",
            "updatedAt",
        )
        created_at = self._parse_datetime(created_text)
        updated_at = self._parse_datetime(updated_text)
        now = datetime.now(timezone.utc)
        raindrop_id = self._first(item, "_id", "id", "item_id")

        return KnowledgeUnit(
            source_project=SourceProject.RAINDROP_JSON,
            source_id=self._source_id(raindrop_id, url),
            source_entity_type="bookmark",
            title=title,
            content=self._content(title, url, description, collection, tags, notes),
            content_type=ContentType.ARTIFACT,
            metadata={
                "url": url,
                "description": description,
                "excerpt": description,
                "collection": collection,
                "notes": notes,
                "created_at": created_text,
                "updated_at": updated_text,
                "tags": tags,
                "raindrop_id": raindrop_id,
                "type": self._first(item, "type"),
                "domain": self._first(item, "domain"),
                "cover": self._first(item, "cover"),
                "source_file": source_file,
            },
            tags=tags,
            created_at=created_at or updated_at or now,
            updated_at=updated_at or created_at or now,
        )

    def _has_bookmark_shape(self, value: dict[str, Any]) -> bool:
        return any(key in value for key in ("link", "url", "href"))

    def _source_id(self, raindrop_id: str, url: str) -> str:
        if raindrop_id:
            return f"raindrop_json:{raindrop_id}"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return f"url:{digest[:24]}"

    def _content(
        self,
        title: str,
        url: str,
        description: str,
        collection: str,
        tags: list[str],
        notes: str,
    ) -> str:
        parts = [title, f"URL: {url}"]
        if description:
            parts.append(f"Description: {description}")
        if notes:
            parts.append(f"Notes: {notes}")
        if collection:
            parts.append(f"Collection: {collection}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _collection(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._first(value, "title", "name", "path", "$id", "id")
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    text = self._first(item, "title", "name", "path", "$id", "id")
                else:
                    text = str(item).strip()
                if text:
                    parts.append(text)
            return " / ".join(parts)
        if value is None:
            return ""
        return str(value).strip()

    def _notes(self, value: Any) -> str:
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value).strip()
        if isinstance(value, dict):
            return self._first(value, "text", "body", "content", "note")
        if isinstance(value, list):
            notes = []
            for item in value:
                if isinstance(item, dict):
                    text = self._first(item, "text", "body", "content", "note")
                else:
                    text = str(item).strip()
                if text:
                    notes.append(re.sub(r"\s+", " ", text).strip())
            return "\n".join(notes)
        return ""

    def _tags(self, value: Any) -> list[str]:
        if isinstance(value, dict):
            raw_tags = value.values()
        elif isinstance(value, list):
            raw_tags = value
        elif isinstance(value, str):
            raw_tags = re.split(r"[,;|]", value)
        else:
            raw_tags = []

        tags: set[str] = set()
        for tag in raw_tags:
            if isinstance(tag, dict):
                tag = tag.get("tag") or tag.get("name") or tag.get("title") or ""
            normalized = re.sub(r"\s+", " ", str(tag).strip().removeprefix("#")).strip().lower()
            if normalized:
                tags.add(normalized)
        return sorted(tags)

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is None or isinstance(value, (dict, list)):
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
