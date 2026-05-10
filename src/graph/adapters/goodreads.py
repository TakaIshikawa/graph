"""Adapter for Goodreads reading history and reviews exports."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GoodreadsAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "goodreads"

    @property
    def entity_types(self) -> list[str]:
        return ["book"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "book" not in entity_types:
            return result

        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists() or not path.is_file():
            return result

        try:
            items = self._read_items(path)
        except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for item in items:
            title = self._title(item)
            author = self._author(item)
            if not title and not author:
                continue

            # Use Date Read as the primary timestamp, fallback to Date Added
            date_read = self._parse_datetime(self._first(item, "Date Read", "date_read"))
            date_added = self._parse_datetime(self._first(item, "Date Added", "date_added"))
            comparable_at = date_read or date_added
            if sync_at and comparable_at and comparable_at <= sync_at:
                continue

            isbn = self._first(item, "ISBN13", "ISBN", "isbn13", "isbn")
            book_id = self._first(item, "Book Id", "book_id", "id")
            my_rating = self._first(item, "My Rating", "my_rating", "rating")
            avg_rating = self._first(item, "Average Rating", "average_rating")
            publisher = self._first(item, "Publisher", "publisher")
            year_published = self._first(
                item, "Year Published", "Original Publication Year", "year_published"
            )
            shelves = self._shelves(item)
            exclusive_shelf = self._first(item, "Exclusive Shelf", "exclusive_shelf")
            review = self._first(item, "My Review", "my_review", "review")
            read_count = self._first(item, "Read Count", "read_count")

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GOODREADS,
                    source_id=self._source_id(book_id, isbn, title, author),
                    source_entity_type="book",
                    title=self._format_title(title, author),
                    content=self._content(title, author, review, shelves, my_rating),
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "book_id": book_id,
                        "title": title,
                        "author": author,
                        "isbn": isbn,
                        "my_rating": my_rating,
                        "average_rating": avg_rating,
                        "publisher": publisher,
                        "year_published": year_published,
                        "date_read": self._first(item, "Date Read", "date_read"),
                        "date_added": self._first(item, "Date Added", "date_added"),
                        "shelves": shelves,
                        "exclusive_shelf": exclusive_shelf,
                        "review": review,
                        "read_count": read_count,
                    },
                    tags=shelves,
                    created_at=date_added or date_read or datetime.now(timezone.utc),
                    updated_at=date_read or date_added or datetime.now(timezone.utc),
                )
            )

        return result

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                return [
                    {str(key).strip(): value for key, value in row.items() if key is not None}
                    for row in csv.DictReader(handle)
                ]

        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._json_items(parsed)

    def _json_items(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []

        for key in ("books", "items", "reviews"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                return [item for item in nested.values() if isinstance(item, dict)]

        if any(isinstance(item, dict) for item in value.values()):
            return [item for item in value.values() if isinstance(item, dict)]
        return [value]

    def _title(self, item: dict[str, Any]) -> str:
        return self._first(item, "Title", "title", "book_title")

    def _author(self, item: dict[str, Any]) -> str:
        return self._first(item, "Author", "author", "book_author")

    def _source_id(self, book_id: str, isbn: str, title: str, author: str) -> str:
        if book_id:
            return f"goodreads:{book_id}"
        if isbn:
            return f"isbn:{isbn}"
        identifier = f"{title}|{author}" if author else title
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
        return f"goodreads:{digest[:24]}"

    def _format_title(self, title: str, author: str) -> str:
        if title and author:
            return f"{title} by {author}"
        return title or author or "Untitled Goodreads book"

    def _content(
        self, title: str, author: str, review: str, shelves: list[str], rating: str
    ) -> str:
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if author:
            parts.append(f"Author: {author}")
        if rating:
            parts.append(f"Rating: {rating}/5")
        if shelves:
            parts.append(f"Shelves: {', '.join(shelves)}")
        if review:
            parts.append(f"\nReview:\n{review}")
        return "\n".join(parts)

    def _shelves(self, item: dict[str, Any]) -> list[str]:
        # Goodreads exports shelves in a "Bookshelves" column, comma-separated
        shelves_str = self._first(item, "Bookshelves", "bookshelves", "shelves")
        if not shelves_str:
            return []

        shelves: list[str] = []
        for shelf in re.split(r",", shelves_str):
            normalized = shelf.strip().lower()
            if normalized and normalized not in shelves:
                shelves.append(normalized)
        return shelves

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        # Handle Unix timestamp
        if re.fullmatch(r"\d+(?:\.0+)?", value):
            try:
                return datetime.fromtimestamp(int(float(value)), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        # Handle ISO format and Goodreads format (YYYY/MM/DD)
        try:
            # Try ISO format first
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            # Try Goodreads format YYYY/MM/DD
            try:
                parsed = datetime.strptime(value, "%Y/%m/%d")
            except ValueError:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
