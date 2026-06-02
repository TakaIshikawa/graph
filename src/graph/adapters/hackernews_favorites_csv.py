"""Adapter for Hacker News favorites CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class HackerNewsFavoritesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "hackernews_favorites_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["favorite"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "favorite" not in entity_types:
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
        item_id = first(row, "item_id", "id")
        title = first(row, "title") or f"Hacker News item {item_id or index + 1}"
        url = first(row, "url")
        author = first(row, "author", "by")
        item_type = first(row, "type") or "item"
        created_text = first(row, "created_at", "created", "time")
        text = first(row, "text", "comment")
        if not any([item_id, title, url, text]):
            return None
        created_at = parse_datetime(created_text) or datetime.now(timezone.utc)
        metadata = clean_metadata({"item_id": item_id, "url": url, "author": author, "type": item_type, "created_at": created_text, "source_file": source_file})
        return KnowledgeUnit(
            source_project="hackernews_favorites_csv",
            source_id=digest_source_id("hackernews_favorites_csv", item_id or url or title or index),
            source_entity_type="favorite",
            title=title,
            content=_content(title, text, url),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            created_at=created_at,
            updated_at=created_at,
        )


def _content(title: str, text: str, url: str) -> str:
    parts = [title, text if text != title else "", f"URL: {url}" if url else ""]
    return "\n".join(part for part in parts if part)
