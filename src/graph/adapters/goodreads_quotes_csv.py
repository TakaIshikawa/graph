"""Adapter for Goodreads quote CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_int, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GoodreadsQuotesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "goodreads_quotes_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["quote"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "quote" not in entity_types:
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
        quote = first(row, "Quote", "Text", "Body")
        if not quote:
            return None
        author = first(row, "Author", "Quote Author")
        book = first(row, "Book", "Title", "Work")
        tags = split_values(first(row, "Tags", "Tag", "Shelves"))
        added = parse_datetime(first(row, "Date Added", "Added", "Created At"))
        now = datetime.now(timezone.utc)
        created_at = added or now
        page = parse_int(first(row, "Page", "Page Number"))
        url = first(row, "URL", "Link")
        quote_id = first(row, "Quote ID", "ID", "id")
        metadata = clean_metadata(
            {
                "quote": quote,
                "author": author,
                "book": book,
                "tags": tags,
                "date_added": added.isoformat() if added else first(row, "Date Added", "Added", "Created At"),
                "page": page,
                "url": url,
                "quote_id": quote_id,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project="goodreads_quotes_csv",
            source_id=self._source_id(quote_id, quote, author, book, page, url, index),
            source_entity_type="quote",
            title=self._title(author, book),
            content=self._content(quote, author, book, page, url),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=["goodreads", *[tag.casefold() for tag in tags]],
            created_at=created_at,
            updated_at=created_at,
        )

    def _source_id(self, quote_id: str, quote: str, author: str, book: str, page: int | None, url: str, index: int) -> str:
        if quote_id:
            return f"goodreads_quotes_csv:{quote_id}"
        return digest_source_id("goodreads_quotes_csv", quote, author, book, page, url, index if not url else "")

    def _title(self, author: str, book: str) -> str:
        if author and book:
            return f"Quote by {author} from {book}"
        if author:
            return f"Quote by {author}"
        return book or "Goodreads quote"

    def _content(self, quote: str, author: str, book: str, page: int | None, url: str) -> str:
        parts = [quote]
        if author:
            parts.append(f"Author: {author}")
        if book:
            parts.append(f"Book: {book}")
        if page is not None:
            parts.append(f"Page: {page}")
        if url:
            parts.append(f"URL: {url}")
        return "\n".join(parts)
