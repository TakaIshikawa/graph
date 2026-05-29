"""Adapter for Pinboard bookmark CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PinboardBookmarksCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pinboard_bookmarks_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "bookmark" not in entity_types:
            return result
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit:
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        url = first(row, "href", "url")
        title = first(row, "description", "title")
        if not url and not title:
            return None
        notes = first(row, "extended", "notes", "note")
        tags = _tags(first(row, "tags", "tag"))
        timestamp = first(row, "time", "created", "date")
        created_at = parse_datetime(timestamp) or datetime.now(timezone.utc)
        shared = _bool(first(row, "shared"))
        toread = _bool(first(row, "toread", "to read"))
        private = _bool(first(row, "private"))
        metadata = clean_metadata({"url": url, "external_url": url, "source_url": url, "notes": notes, "tags": tags, "time": timestamp, "shared": shared, "toread": toread, "private": private, "source_file": source_file})
        return KnowledgeUnit(source_project="pinboard_bookmarks_csv", source_id=digest_source_id("pinboard_bookmarks_csv", url or title or index), source_entity_type="bookmark", title=title or url or "Pinboard bookmark", content=_content(title, url, notes, tags), content_type=ContentType.ARTIFACT, metadata=metadata, tags=tags, created_at=created_at, updated_at=created_at)


def _tags(value: str) -> list[str]:
    tags: list[str] = []
    for item in value.replace(",", " ").split():
        tag = item.strip().casefold()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _bool(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y", "on"}


def _content(title: str, url: str, notes: str, tags: list[str]) -> str:
    return "\n".join(part for part in (title, f"URL: {url}" if url else "", notes, f"Tags: {', '.join(tags)}" if tags else "") if part)
