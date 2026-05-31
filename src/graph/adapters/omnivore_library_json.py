"""Adapter for Omnivore library JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class OmnivoreLibraryJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "omnivore_library_json"

    @property
    def entity_types(self) -> list[str]:
        return ["bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types and "bookmark" not in entity_types:
            return result
        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.is_file():
            return result
        try:
            items = self._items(json.loads(path.read_text(encoding="utf-8-sig")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return result
        sync_at = self._sync_datetime(since) if since else None
        now = datetime.now(timezone.utc)
        for item in items:
            title = self._text(item, "title", "name") or "Untitled Omnivore bookmark"
            url = self._text(item, "url", "originalUrl", "original_url")
            slug = self._text(item, "slug", "id", "pageId", "page_id")
            saved_at = self._date(self._text(item, "savedAt", "saved_at", "createdAt", "created_at")) or now
            updated_at = self._date(self._text(item, "updatedAt", "updated_at")) or saved_at
            if sync_at and updated_at <= sync_at:
                continue
            labels = self._labels(item)
            highlights = self._highlights(item)
            source_id = f"omnivore_library_json:bookmark:{slug or self._digest(url, title)}"
            content = "\n".join(part for part in [title, url, self._text(item, "description"), *highlights] if part)
            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.OMNIVORE_JSON,
                    source_id=source_id,
                    source_entity_type="bookmark",
                    title=title,
                    content=content,
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "url": url,
                        "slug": slug,
                        "state": self._text(item, "state", "status"),
                        "description": self._text(item, "description"),
                        "labels": labels,
                        "highlights": highlights,
                        "highlight_count": len(highlights),
                        "saved_at": saved_at.isoformat(),
                        "updated_at": updated_at.isoformat(),
                    },
                    tags=labels,
                    created_at=saved_at,
                    updated_at=updated_at,
                )
            )
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _items(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []
        for key in ("items", "nodes", "edges", "data", "pages"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [self._node(item) for item in nested if isinstance(self._node(item), dict)]
            if isinstance(nested, dict):
                found = self._items(nested)
                if found:
                    return found
        return [value] if self._text(value, "title", "url") else []

    def _node(self, item: dict[str, Any]) -> Any:
        return item.get("node") if isinstance(item.get("node"), dict) else item

    def _labels(self, item: dict[str, Any]) -> list[str]:
        raw = item.get("labels") or item.get("tags") or []
        values = raw if isinstance(raw, list) else [raw]
        labels: list[str] = []
        for value in values:
            text = self._text(value, "name", "label") if isinstance(value, dict) else str(value)
            tag = text.strip().lower()
            if tag and tag not in labels:
                labels.append(tag)
        return labels

    def _highlights(self, item: dict[str, Any]) -> list[str]:
        raw = item.get("highlights") or []
        values = raw if isinstance(raw, list) else [raw]
        highlights: list[str] = []
        for value in values:
            text = self._text(value, "quote", "text", "highlight") if isinstance(value, dict) else str(value)
            if text.strip():
                highlights.append(text.strip())
        return highlights

    def _text(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value not in (None, ""):
                return str(value).strip()
        return ""

    def _date(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _digest(self, *values: str) -> str:
        return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()[:16]

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = getattr(since, "last_sync_at", None)
        return value if value and value.tzinfo else (value.replace(tzinfo=timezone.utc) if value else datetime.min.replace(tzinfo=timezone.utc))
