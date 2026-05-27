"""Adapter for Pocket article CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class PocketArticlesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pocket_articles_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["article"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "article" not in entity_types:
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
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        url = first(row, "url", "href", "given_url", "resolved_url")
        title = first(row, "title", "resolved_title", "given_title")
        excerpt = first(row, "excerpt", "description")
        if not any((url, title, excerpt)):
            return None
        added_at = parse_datetime(first(row, "time added", "time_added", "added", "created_at")) or datetime.now(timezone.utc)
        read_at = parse_datetime(first(row, "time read", "time_read", "read_at"))
        tags = [tag.casefold() for tag in split_values(first(row, "tags", "tag"))]
        metadata = clean_metadata(
            {
                "url": url,
                "title": title,
                "tags": tags,
                "excerpt": excerpt,
                "time_added": added_at.isoformat(),
                "time_read": read_at.isoformat() if read_at else None,
                "favorite": _truthy(first(row, "favorite", "is favorite")),
                "archived": _truthy(first(row, "archive", "archived", "status")),
                "status": first(row, "status"),
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.POCKET_ARTICLES_CSV,
            source_id=digest_source_id("pocket_articles_csv", url or title, index),
            source_entity_type="article",
            title=title or url or "Pocket article",
            content="\n".join(part for part in (title, url, excerpt) if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=added_at,
            updated_at=read_at or added_at,
        )


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y", "archive", "archived", "favorite"}
