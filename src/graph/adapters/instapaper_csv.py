"""Adapter for Instapaper CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class InstapaperCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "instapaper_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["article"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "article" not in entity_types:
            return result

        sync_at = self._sync_at(since)
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

        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Article Title", "Name")
        url = first(row, "URL", "Url", "Article URL", "Link")
        selection = first(row, "Selection", "Highlight", "Highlighted Text", "Text", "Quote")
        description = first(row, "Description", "Excerpt", "Summary")
        if not any((title, url, selection, description)):
            return None

        folder = first(row, "Folder", "Folder Name", "Folder/State")
        state = first(row, "State", "Status")
        saved_text = first(row, "Date Saved", "Saved At", "Created At", "Date", "Time")
        saved_at = parse_datetime(saved_text)
        progress = first(row, "Progress", "Reading Progress", "Progress Percent", "Progress Percentage")
        progress_position = first(row, "Progress Position", "Position", "Location")
        progress_total = first(row, "Progress Total", "Total", "Length")
        now = datetime.now(timezone.utc)
        tags = self._tags(folder, state)

        return KnowledgeUnit(
            source_project="instapaper_csv",
            source_id=digest_source_id("instapaper_csv", url or title, selection, saved_text, index if not url else ""),
            source_entity_type="article",
            title=title or url or "Instapaper article",
            content=self._content(title, url, folder, state, description, selection, saved_at, progress, progress_position, progress_total),
            content_type=ContentType.ARTIFACT,
            metadata=clean_metadata(
                {
                    "title": title,
                    "url": url,
                    "folder": folder,
                    "state": state,
                    "folder_tag": self._normalize_tag(folder),
                    "state_tag": self._normalize_tag(state),
                    "selection": selection,
                    "highlight": selection,
                    "description": description,
                    "date_saved": saved_at.isoformat() if saved_at else saved_text,
                    "progress": progress,
                    "progress_position": progress_position,
                    "progress_total": progress_total,
                    "source_file": source_file,
                    "row": dict(row),
                }
            ),
            tags=tags,
            created_at=saved_at or now,
            updated_at=saved_at or now,
        )

    def _content(
        self,
        title: str,
        url: str,
        folder: str,
        state: str,
        description: str,
        selection: str,
        saved_at: datetime | None,
        progress: str,
        progress_position: str,
        progress_total: str,
    ) -> str:
        parts: list[str] = []
        if title:
            parts.append(title)
        if url:
            parts.append(f"URL: {url}")
        if folder:
            parts.append(f"Folder: {folder}")
        if state:
            parts.append(f"State: {state}")
        if saved_at:
            parts.append(f"Date saved: {saved_at.isoformat()}")
        if description:
            parts.append(f"Description: {description}")
        if selection:
            parts.append(f"Selection: {selection}")
        progress_parts = [part for part in (progress, progress_position, progress_total) if part]
        if progress_parts:
            parts.append(f"Progress: {' / '.join(progress_parts)}")
        return "\n".join(parts)

    def _tags(self, folder: str, state: str) -> list[str]:
        tags = ["instapaper"]
        for value in (folder, state):
            tag = self._normalize_tag(value)
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _normalize_tag(self, value: str) -> str:
        return re.sub(r"\s+", "-", value.strip().casefold()) if value else ""

    def _sync_at(self, since: SyncState | None) -> datetime | None:
        if since is None:
            return None
        value = since.last_sync_at
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
