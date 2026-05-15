"""Adapter for Raindrop.io bookmark CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
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
        if sync_at and sync_at.tzinfo is None:
            sync_at = sync_at.replace(tzinfo=timezone.utc)

        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "title", "name")
        url = first(row, "url", "link", "href")
        excerpt = first(row, "excerpt", "description", "summary")
        note = first(row, "note", "notes")
        if not any((title, url, excerpt, note)):
            return None

        collection = first(row, "collection", "folder", "collection_title", "collectionTitle")
        created_text = first(row, "created", "created_at", "createdAt", "date")
        updated_text = first(row, "lastUpdate", "last_update", "lastUpdated", "updated", "updated_at")
        created_at = parse_datetime(created_text)
        updated_at = parse_datetime(updated_text)
        now = datetime.now(timezone.utc)
        tags = self._tags(first(row, "tags", "tag"))
        favorite = self._truthy(first(row, "favorite", "favourite", "is_favorite"))
        broken = self._truthy(first(row, "broken", "is_broken"))

        metadata = clean_metadata(
            {
                "title": title or url,
                "url": url,
                "source_url": url,
                "external_url": url,
                "excerpt": excerpt,
                "note": note,
                "tags": tags,
                "collection": collection,
                "favorite": favorite,
                "broken": broken,
                "created": created_text,
                "created_at": created_at.isoformat() if created_at else None,
                "last_update": updated_text,
                "updated_at": updated_at.isoformat() if updated_at else None,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.RAINDROP_BOOKMARKS_CSV,
            source_id=self._source_id(url, title, index),
            source_entity_type="bookmark",
            title=title or url or "Untitled Raindrop bookmark",
            content=self._content(title, url, excerpt, note, collection, tags, favorite, broken),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at or updated_at or now,
            updated_at=updated_at or created_at or now,
        )

    def _source_id(self, url: str, title: str, index: int) -> str:
        return digest_source_id("raindrop_bookmarks_csv", url or title or index)

    def _content(
        self,
        title: str,
        url: str,
        excerpt: str,
        note: str,
        collection: str,
        tags: list[str],
        favorite: bool,
        broken: bool,
    ) -> str:
        parts = [part for part in (title, f"URL: {url}" if url else "") if part]
        if excerpt:
            parts.append(f"Excerpt: {excerpt}")
        if note:
            parts.append(f"Note: {note}")
        if collection:
            parts.append(f"Collection: {collection}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if favorite:
            parts.append("Favorite: true")
        if broken:
            parts.append("Broken: true")
        return "\n".join(parts)

    def _tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for tag in split_values(value):
            normalized = " ".join(tag.removeprefix("#").casefold().split())
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _truthy(self, value: str) -> bool:
        return value.strip().casefold() in {"1", "true", "yes", "y", "on", "favorite", "favourite"}
