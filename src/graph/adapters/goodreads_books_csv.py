"""Adapter for Goodreads books CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GoodreadsBooksCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "goodreads_books_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["book"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "book" not in entity_types:
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
        book_id = first(row, "Book Id", "book_id", "id")
        title = first(row, "Title")
        author = first(row, "Author", "Authors")
        review = first(row, "My Review", "Review")
        if not any([book_id, title, author, review]):
            return None
        added_text = first(row, "Date Added", "Date Read")
        created_at = parse_datetime(added_text) or datetime.now(timezone.utc)
        bookshelves = split_values(first(row, "Bookshelves", "Exclusive Shelf"))
        metadata = clean_metadata(
            {
                "book_id": book_id,
                "author": author,
                "isbn": first(row, "ISBN", "ISBN13"),
                "my_rating": first(row, "My Rating"),
                "average_rating": first(row, "Average Rating"),
                "bookshelves": bookshelves,
                "date_read": first(row, "Date Read"),
                "date_added": first(row, "Date Added"),
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="goodreads_books_csv",
            source_id=digest_source_id("goodreads_books_csv", book_id or title or index),
            source_entity_type="book",
            title=title or f"Goodreads book {book_id or index + 1}",
            content=_content(title, author, review, bookshelves),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=bookshelves,
            created_at=created_at,
            updated_at=created_at,
        )


def _content(title: str, author: str, review: str, bookshelves: list[str]) -> str:
    parts = [title, f"Author: {author}" if author else "", review, f"Bookshelves: {', '.join(bookshelves)}" if bookshelves else ""]
    return "\n".join(part for part in parts if part)
