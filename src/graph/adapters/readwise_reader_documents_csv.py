"""Adapter for Readwise Reader document CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class ReadwiseReaderDocumentsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "readwise_reader_documents_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["reader_document"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "reader_document" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        units_by_id: dict[str, KnowledgeUnit] = {}

        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                previous = units_by_id.get(unit.source_id)
                if previous is None or self._dedupe_key(unit) > self._dedupe_key(previous):
                    units_by_id[unit.source_id] = unit

        result.units = sorted(units_by_id.values(), key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        document_id = first(row, "Document ID", "Document Id", "ID", "Id", "Reader ID")
        title = first(row, "Title", "Document", "Document Title")
        url = first(row, "URL", "Url", "Document URL", "Source URL")
        author = first(row, "Author", "Authors")
        source = first(row, "Source", "Site", "Feed", "Publisher")
        category = first(row, "Category", "Type", "Document Type")
        location = first(row, "Location", "Folder", "List")
        tags = self._tags(first(row, "Tags", "Tag"))
        saved_text = first(row, "Saved At", "Saved", "Created At", "Created")
        opened_text = first(row, "Last Opened At", "Last Opened", "Opened At", "Last Read At")
        updated_text = first(row, "Updated At", "Updated", "Modified At", "Modified")
        progress = parse_float(first(row, "Reading Progress", "Progress", "Percent Read"))
        saved_at = parse_datetime(saved_text)
        opened_at = parse_datetime(opened_text)
        updated_at = parse_datetime(updated_text)

        if not any([document_id, title, url, author, source, category, location, tags, saved_text, opened_text, updated_text]):
            return None

        now = datetime.now(timezone.utc)
        created_at = saved_at or opened_at or updated_at or now
        modified_at = updated_at or opened_at or saved_at or now
        metadata = clean_metadata(
            {
                "document_id": document_id,
                "title": title,
                "url": url,
                "source_url": url,
                "external_url": url,
                "author": author,
                "source": source,
                "category": category,
                "location": location,
                "tags": tags,
                "saved_at": saved_at.isoformat() if saved_at else saved_text,
                "last_opened_at": opened_at.isoformat() if opened_at else opened_text,
                "updated_at": updated_at.isoformat() if updated_at else updated_text,
                "reading_progress": progress,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="readwise_reader_documents_csv",
            source_id=self._source_id(document_id, url, title, index),
            source_entity_type="reader_document",
            title=title or url or "Readwise Reader document",
            content=self._content(title, url, author, source, category, location, tags, progress),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at,
            updated_at=modified_at,
        )

    def _source_id(self, document_id: str, url: str, title: str, index: int) -> str:
        if document_id:
            return f"readwise_reader_documents_csv:{document_id}"
        return digest_source_id("readwise_reader_documents_csv", url, title, index if not (url or title) else "")

    def _tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for tag in split_values(value):
            normalized = tag.strip().removeprefix("#")
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _content(
        self,
        title: str,
        url: str,
        author: str,
        source: str,
        category: str,
        location: str,
        tags: list[str],
        progress: float | None,
    ) -> str:
        parts: list[str] = []
        if title:
            parts.append(title)
        if author:
            parts.append(f"Author: {author}")
        if source:
            parts.append(f"Source: {source}")
        if category:
            parts.append(f"Category: {category}")
        if location:
            parts.append(f"Location: {location}")
        if url:
            parts.append(f"URL: {url}")
        if progress is not None:
            parts.append(f"Reading Progress: {progress:g}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _dedupe_key(self, unit: KnowledgeUnit) -> tuple[datetime, datetime, str, str]:
        return (unit.updated_at, unit.created_at, str(unit.metadata.get("source_file") or ""), unit.title)
