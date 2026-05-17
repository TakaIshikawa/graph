"""Adapter for Readwise Reader highlight CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class ReadwiseReaderHighlightsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "readwise_reader_highlights_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["highlight"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "highlight" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

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
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        highlight = first(row, "Highlight", "Text", "Highlighted Text", "Quote")
        note = first(row, "Note", "Notes", "Annotation")
        document = first(row, "Document", "Document Title", "Title", "Article Title", "Book Title")
        author = first(row, "Author", "Authors", "Book Author")
        url = first(row, "URL", "Source URL", "Document URL", "Article URL", "Book URL")
        location = first(row, "Location", "Position")
        tags = self._tags(first(row, "Tags", "Tag"))
        highlighted_at_text = first(row, "Highlighted At", "Highlighted at", "Created At", "Created", "Date")
        updated_at_text = first(row, "Updated At", "Updated at", "Modified At", "Modified")
        highlighted_at = parse_datetime(highlighted_at_text)
        updated_at = parse_datetime(updated_at_text)

        if not any([highlight, note, document, author, url, location, tags, highlighted_at_text, updated_at_text]):
            return None

        now = datetime.now(timezone.utc)
        created_at = highlighted_at or updated_at or now
        modified_at = updated_at or highlighted_at or now
        metadata = clean_metadata(
            {
                "highlight": highlight,
                "document": document,
                "author": author,
                "url": url,
                "note": note,
                "tags": tags,
                "location": location,
                "highlighted_at": highlighted_at.isoformat() if highlighted_at else highlighted_at_text,
                "updated_at": updated_at.isoformat() if updated_at else updated_at_text,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )

        return KnowledgeUnit(
            source_project="readwise_reader_highlights_csv",
            source_id=self._source_id(row, document, author, url, highlight, note, location, highlighted_at_text, index),
            source_entity_type="highlight",
            title=document or "Readwise Reader highlight",
            content=self._content(highlight, note, document, author, url, location, tags),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=tags,
            created_at=created_at,
            updated_at=modified_at,
        )

    def _source_id(
        self,
        row: dict[str, Any],
        document: str,
        author: str,
        url: str,
        highlight: str,
        note: str,
        location: str,
        highlighted_at: str,
        index: int,
    ) -> str:
        highlight_id = first(row, "Highlight ID", "Highlight Id", "ID", "Id", "Readwise ID")
        if highlight_id:
            return f"readwise_reader_highlights_csv:{highlight_id}"
        return digest_source_id("readwise_reader_highlights_csv", document, author, url, highlight, note, location, highlighted_at, index if not highlight else "")

    def _tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for tag in split_values(value):
            normalized = tag.strip().removeprefix("#")
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _content(self, highlight: str, note: str, document: str, author: str, url: str, location: str, tags: list[str]) -> str:
        parts: list[str] = []
        if highlight:
            parts.append(highlight)
        if note:
            parts.append(f"Note: {note}")
        if document:
            parts.append(f"Document: {document}")
        if author:
            parts.append(f"Author: {author}")
        if url:
            parts.append(f"URL: {url}")
        if location:
            parts.append(f"Location: {location}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)
