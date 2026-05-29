"""Adapter for Pocket bookmarks CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PocketBookmarksCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pocket_bookmarks_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "bookmark" not in entity_types:
            return result
        sync_at = since.last_sync_at.replace(tzinfo=timezone.utc) if since and since.last_sync_at.tzinfo is None else (since.last_sync_at if since else None)
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "title", "given_title", "resolved_title")
        url = first(row, "url", "href", "given_url", "resolved_url")
        if not title and not url:
            return None
        tags = _tags(first(row, "tags", "tag"))
        added_text = first(row, "time_added", "created_at", "date_added")
        created_at = parse_datetime(added_text) or datetime.now(timezone.utc)
        status = first(row, "status", "state")
        excerpt = first(row, "excerpt", "description", "summary")
        favorite = _bool(first(row, "favorite", "is_favorite", "favorited"))
        metadata = clean_metadata({"url": url, "external_url": url, "source_url": url, "tags": tags, "time_added": added_text, "status": status, "excerpt": excerpt, "favorite": favorite, "source_file": source_file})
        return KnowledgeUnit(source_project="pocket_bookmarks_csv", source_id=digest_source_id("pocket_bookmarks_csv", url or title or index), source_entity_type="bookmark", title=title or url or "Pocket bookmark", content=_content(title, url, excerpt, tags), content_type=ContentType.ARTIFACT, metadata=metadata, tags=tags, created_at=created_at, updated_at=created_at)


def _tags(value: str) -> list[str]:
    tags: list[str] = []
    for tag in split_values(value):
        normalized = " ".join(tag.removeprefix("#").casefold().split())
        if normalized and normalized not in tags:
            tags.append(normalized)
    return tags


def _bool(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y", "on", "favorite", "favorited"}


def _content(title: str, url: str, excerpt: str, tags: list[str]) -> str:
    return "\n".join(part for part in (title, f"URL: {url}" if url else "", excerpt, f"Tags: {', '.join(tags)}" if tags else "") if part)
