"""Adapter for Raindrop.io bookmark CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class RaindropBookmarksCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "raindrop_bookmarks_csv"

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
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "title", "name")
        url = first(row, "link", "url", "href")
        if not title and not url:
            return None
        excerpt = first(row, "excerpt", "description")
        note = first(row, "note", "notes")
        tags = _tags(first(row, "tags", "tag"))
        collection = first(row, "collection", "folder")
        favorite = _bool(first(row, "favorite", "important"))
        broken = _bool(first(row, "broken"))
        created_text = first(row, "created", "created_at", "date")
        updated_text = first(row, "lastUpdate", "last_update", "updated", "updated_at", "modified")
        created_at = parse_datetime(created_text) or datetime.now(timezone.utc)
        updated_at = parse_datetime(updated_text) or created_at
        metadata = clean_metadata(
            {
                "title": title,
                "url": url,
                "external_url": url,
                "source_url": url,
                "excerpt": excerpt,
                "note": note,
                "tags": tags,
                "collection": collection,
                "favorite": favorite,
                "broken": broken,
                "created": created_text,
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="raindrop_bookmarks_csv",
            source_id=digest_source_id("raindrop_bookmarks_csv", url or title or index),
            source_entity_type="bookmark",
            title=title or url or "Raindrop bookmark",
            content=_content(title, url, excerpt, note, tags, collection),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at,
            updated_at=updated_at,
        )


def _tags(value: str) -> list[str]:
    tags: list[str] = []
    for tag in split_values(value):
        normalized = tag.lstrip("#").casefold()
        if normalized and normalized not in tags:
            tags.append(normalized)
    return tags


def _bool(value: str) -> bool | None:
    text = value.strip().casefold()
    if not text:
        return None
    return text in {"1", "true", "yes", "y", "on"}


def _content(title: str, url: str, excerpt: str, note: str, tags: list[str], collection: str) -> str:
    parts: list[str] = []
    for part in (
        title,
        f"URL: {url}" if url else "",
        excerpt,
        note if note != excerpt else "",
        f"Excerpt: {excerpt}" if excerpt else "",
        f"Note: {note}" if note and note != excerpt else "",
        f"Collection: {collection}" if collection else "",
        f"Tags: {', '.join(tags)}" if tags else "",
    ):
        if part:
            parts.append(part)
    return "\n".join(parts)
