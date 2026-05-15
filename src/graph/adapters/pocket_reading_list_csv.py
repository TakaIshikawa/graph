"""Adapter for Pocket reading list CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class PocketReadingListCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pocket_reading_list_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["saved_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "saved_item" not in entity_types:
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
        url = first(row, "url", "given_url", "resolved_url", "item_url", "link")
        title = first(row, "title", "given_title", "resolved_title", "item_title")
        if not url and not title:
            return None

        added_text = first(row, "time_added", "added_at", "created_at", "date_added")
        read_text = first(row, "time_read", "read_at", "date_read")
        updated_text = first(row, "time_updated", "updated_at", "modified_at")
        added_at = parse_datetime(added_text)
        read_at = parse_datetime(read_text)
        updated_at = parse_datetime(updated_text) or read_at
        status = self._status(row)
        archived = status in {"archive", "archived", "read"} or self._truthy(first(row, "archived", "is_archived"))
        favorite = self._truthy(first(row, "favorite", "is_favorite", "favorited"))
        tags = self._tags(first(row, "tags", "tag"))
        excerpt = first(row, "excerpt", "resolved_excerpt", "description", "summary")
        language = first(row, "lang", "language")
        domain = self._domain(url)
        now = datetime.now(timezone.utc)

        metadata = clean_metadata(
            {
                "title": title or url,
                "url": url,
                "source_url": url,
                "external_url": url,
                "time_added": added_text,
                "time_read": read_text,
                "added_at": added_at.isoformat() if added_at else None,
                "read_at": read_at.isoformat() if read_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
                "status": status,
                "archived": archived,
                "favorite": favorite,
                "tags": tags,
                "excerpt": excerpt,
                "language": language,
                "domain": domain,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.POCKET_READING_LIST_CSV,
            source_id=self._source_id(url, title, index),
            source_entity_type="saved_item",
            title=title or url or "Untitled Pocket item",
            content=self._content(title, url, status, archived, favorite, tags, excerpt, language),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=added_at or updated_at or now,
            updated_at=updated_at or added_at or now,
        )

    def _source_id(self, url: str, title: str, index: int) -> str:
        return digest_source_id("pocket_reading_list_csv", url or title or index)

    def _status(self, row: dict[str, Any]) -> str:
        status = first(row, "status", "state")
        if status:
            return status.casefold().replace(" ", "_")
        if first(row, "time_read", "read_at", "date_read"):
            return "read"
        return "unread"

    def _tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for tag in split_values(value):
            normalized = " ".join(tag.removeprefix("#").casefold().split())
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _truthy(self, value: str) -> bool:
        return value.strip().casefold() in {"1", "true", "yes", "y", "on", "favorite", "favorited"}

    def _domain(self, url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url if "://" in url else f"https://{url}")
        return (parsed.netloc or parsed.path).casefold().removeprefix("www.")

    def _content(
        self,
        title: str,
        url: str,
        status: str,
        archived: bool,
        favorite: bool,
        tags: list[str],
        excerpt: str,
        language: str,
    ) -> str:
        parts = [part for part in (title, f"URL: {url}" if url else "") if part]
        if status:
            parts.append(f"Status: {status}")
        if archived:
            parts.append("Archived: true")
        if favorite:
            parts.append("Favorite: true")
        if language:
            parts.append(f"Language: {language}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if excerpt:
            parts.append(f"Excerpt: {excerpt}")
        return "\n".join(parts)
