"""Adapter for Apple Books highlights CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AppleBooksHighlightsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "apple_books_highlights_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["highlight"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "highlight" not in entity_types:
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
        highlight = first(row, "Highlight", "Highlighted Text", "Selected Text", "Selection", "Text", "Quote")
        note = first(row, "Note", "Notes", "Annotation", "Comment")
        if not highlight and not note:
            return None

        book_title = first(row, "Book Title", "Title", "Book")
        author = first(row, "Author", "Authors")
        chapter = first(row, "Chapter", "Chapter Title", "Section")
        page = first(row, "Page", "Page Number")
        location = first(row, "Location", "Location in Book", "Position", "Cfi", "CFI", "Range") or page
        color = first(row, "Color", "Colour", "Highlight Color")
        created_text = first(row, "Created Date", "Date Created", "Created", "Created At", "Date")
        created_at = parse_datetime(created_text)
        source_identifier = first(row, "Source Identifier", "Source ID", "Persistent ID", "Asset ID", "ZUUID", "ID")
        now = datetime.now(timezone.utc)

        return KnowledgeUnit(
            source_project="apple_books_highlights_csv",
            source_id=self._source_id(source_identifier, book_title, author, chapter, location, page, highlight, note, created_text, index),
            source_entity_type="highlight",
            title=self._title(book_title, highlight or note),
            content=self._content(book_title, author, chapter, location, page, color, highlight, note),
            content_type=ContentType.INSIGHT,
            metadata=clean_metadata(
                {
                    "book_title": book_title,
                    "author": author,
                    "highlight": highlight,
                    "note": note,
                    "chapter": chapter,
                    "location": location,
                    "page": page,
                    "color": color,
                    "created_date": created_at.isoformat() if created_at else created_text,
                    "source_identifier": source_identifier,
                    "source_file": source_file,
                    "row": dict(row),
                }
            ),
            tags=self._tags(color),
            created_at=created_at or now,
            updated_at=created_at or now,
        )

    def _source_id(
        self,
        source_identifier: str,
        book_title: str,
        author: str,
        chapter: str,
        location: str,
        page: str,
        highlight: str,
        note: str,
        created_text: str,
        index: int,
    ) -> str:
        if source_identifier:
            return f"apple_books_highlights_csv:{source_identifier}"
        return digest_source_id("apple_books_highlights_csv", book_title, author, chapter, location, page, highlight, note, created_text, index)

    def _title(self, book_title: str, text: str) -> str:
        if book_title:
            return f"Apple Books highlight: {book_title}"
        return f"Apple Books highlight: {text[:80]}" if text else "Apple Books highlight"

    def _content(self, book_title: str, author: str, chapter: str, location: str, page: str, color: str, highlight: str, note: str) -> str:
        parts: list[str] = []
        if book_title:
            parts.append(f"Book: {book_title}")
        if author:
            parts.append(f"Author: {author}")
        if chapter:
            parts.append(f"Chapter: {chapter}")
        if page and page != location:
            parts.append(f"Page: {page}")
        if location:
            parts.append(f"Location: {location}")
        if color:
            parts.append(f"Color: {color}")
        if highlight:
            parts.append(f"Highlight: {highlight}")
        if note:
            parts.append(f"Note: {note}")
        return "\n".join(parts)

    def _tags(self, color: str) -> list[str]:
        tags = ["apple-books", "highlight"]
        if color:
            tags.append(color.strip().casefold())
        return tags

    def _sync_at(self, since: SyncState | None) -> datetime | None:
        if since is None:
            return None
        value = since.last_sync_at
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
