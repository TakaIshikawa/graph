"""Adapter for Apple Books library CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AppleBooksLibraryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "apple_books_library_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["book"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "book" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows, start=2):
                unit = self._unit(row, path.name, index)
                if unit is None or (sync_at and unit.updated_at <= sync_at):
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, row: dict[str, Any], source_file: str, source_row: int) -> KnowledgeUnit | None:
        title = first(row, "Title", "Book Title", "Name")
        author = first(row, "Author", "Authors")
        isbn = first(row, "ISBN", "ISBN13", "ISBN-13")
        book_id = first(row, "Book ID", "Apple ID", "Store ID", "ID")
        if not any([title, author, isbn, book_id]):
            return None
        pub = parse_datetime(first(row, "Publication Date", "Published", "Release Date"))
        opened = parse_datetime(first(row, "Last Opened", "Last Read", "Last Opened Date"))
        url = first(row, "Store URL", "URL", "Link")
        metadata = clean_metadata({"title": title, "author": author, "sort_author": first(row, "Sort Author", "Author Sort"), "genre": first(row, "Genre", "Category"), "publisher": first(row, "Publisher"), "publication_date": pub.date().isoformat() if pub else first(row, "Publication Date", "Published"), "isbn": isbn, "book_id": book_id, "reading_status": first(row, "Reading Status", "Status"), "percent_complete": parse_float(first(row, "Percent Complete", "% Complete", "Progress")), "last_opened": opened.isoformat() if opened else first(row, "Last Opened", "Last Read"), "url": url, "source_url": url, "external_url": url, "source_file": source_file, "source_row": source_row})
        now = datetime.now(timezone.utc)
        source_id = f"{self.name}:isbn:{isbn}" if isbn else f"{self.name}:book:{book_id}" if book_id else digest_source_id(self.name, title, author)
        return KnowledgeUnit(source_project=self.name, source_id=source_id, source_entity_type="book", title=title or "Apple Books item", content=self._content(title, author, metadata), content_type=ContentType.METADATA, metadata=metadata, tags=list(dict.fromkeys(tag for tag in ["apple_books", "book", metadata.get("genre"), metadata.get("reading_status")] if tag)), created_at=pub or now, updated_at=opened or pub or now)

    def _content(self, title: str, author: str, metadata: dict[str, Any]) -> str:
        parts = [title or "Apple Books item", f"Author: {author}" if author else ""]
        for key, label in (("genre", "Genre"), ("publisher", "Publisher"), ("reading_status", "Status"), ("percent_complete", "Progress"), ("url", "URL")):
            if key in metadata:
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(part for part in parts if part)
