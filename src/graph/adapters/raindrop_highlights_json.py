"""Adapter for Raindrop.io highlight JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class RaindropHighlightsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "raindrop_highlights_json"

    @property
    def entity_types(self) -> list[str]:
        return ["highlight"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "highlight" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._records(json.loads(path.read_text(encoding="utf-8-sig")))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit(record, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [record for item in value for record in self._records(item)]
        if not isinstance(value, dict):
            return []
        if any(key in value for key in ("text", "highlight", "quote")):
            return [value]
        records = []
        for key in ("items", "results", "highlights", "data", "raindrops"):
            if isinstance(value.get(key), (dict, list)):
                records.extend(self._records(value[key]))
        return records

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        text = first(record, "text", "highlight", "quote")
        if not text:
            return None
        bookmark = record.get("bookmark") if isinstance(record.get("bookmark"), dict) else record.get("raindrop") if isinstance(record.get("raindrop"), dict) else {}
        url = first(record, "link", "url", "href") or first(bookmark, "link", "url", "href")
        title = first(record, "title", "bookmarkTitle") or first(bookmark, "title", "name")
        note = first(record, "note", "notes", "annotation")
        collection = self._collection(record.get("collection") or bookmark.get("collection"))
        tags = split_values(record.get("tags") or bookmark.get("tags") or first(record, "tag"))
        color = first(record, "color")
        created_at = parse_datetime(first(record, "created", "createdAt", "created_at", "date")) or datetime.now(timezone.utc)
        metadata = clean_metadata({"highlight": text, "note": note, "url": url, "bookmark_title": title, "collection": collection, "tags": tags, "color": color, "created_at": created_at.isoformat(), "source_file": source_file, "record": dict(record)})
        return KnowledgeUnit(source_project=self.name, source_id=digest_source_id(self.name, first(record, "id", "_id") or url, text, index if not url else ""), source_entity_type="highlight", title=title or text[:80], content=self._content(text, note, title, url, collection, color), content_type=ContentType.ARTIFACT, metadata=metadata, tags=["raindrop", *[tag.casefold() for tag in tags]], created_at=created_at, updated_at=created_at)

    def _collection(self, value: Any) -> str:
        if isinstance(value, dict):
            return first(value, "title", "name", "path", "id")
        if isinstance(value, list):
            return " / ".join(first(item, "title", "name", "path", "id") if isinstance(item, dict) else str(item).strip() for item in value).strip()
        return "" if value is None else str(value).strip()

    def _content(self, text: str, note: str, title: str, url: str, collection: str, color: str) -> str:
        return "\n".join(part for part in (text, f"Note: {note}" if note else "", f"Bookmark: {title}" if title else "", f"URL: {url}" if url else "", f"Collection: {collection}" if collection else "", f"Color: {color}" if color else "") if part)
