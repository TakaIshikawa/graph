"""Adapter for Instapaper bookmark CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class InstapaperBookmarksCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "instapaper_bookmarks_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["saved_article"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "saved_article" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Article Title")
        url = first(row, "URL", "Link", "Article URL")
        if not title and not url:
            return None
        folder = first(row, "Folder", "Category")
        selection = first(row, "Selection", "Highlight", "Selected Text")
        description = first(row, "Description", "Excerpt", "Summary")
        progress = parse_float(first(row, "Progress", "Reading Progress"))
        added = parse_datetime(first(row, "Added", "Time", "Created", "Saved At")) or datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "title": title,
                "url": url,
                "folder": folder,
                "selection": selection,
                "description": description,
                "progress": progress,
                "added": added.isoformat(),
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="instapaper_bookmarks_csv",
            source_id=digest_source_id("instapaper_bookmarks_csv", url or title, "" if url else index),
            source_entity_type="saved_article",
            title=title or url,
            content=self._content(title, url, folder, selection, description, progress),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=[tag for tag in ["instapaper", folder.casefold() if folder else ""] if tag],
            created_at=added,
            updated_at=added,
        )

    def _content(self, title: str, url: str, folder: str, selection: str, description: str, progress: float | None) -> str:
        parts = [part for part in (title, url) if part]
        if folder:
            parts.append(f"Folder: {folder}")
        if progress is not None:
            parts.append(f"Progress: {progress:g}")
        if description:
            parts.append(description)
        if selection:
            parts.append(f"Selection: {selection}")
        return "\n".join(parts)
