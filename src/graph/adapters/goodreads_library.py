"""Adapter for Goodreads library CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class GoodreadsLibraryAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "goodreads_library"

    @property
    def entity_types(self) -> list[str]:
        return ["author", "book", "shelf"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = self._sync_datetime(since) if since else None
        books: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue

            for row in rows:
                title = self._first(row, "Title", "title")
                author = self._first(row, "Author", "author")
                if not title and not author:
                    continue
                date_read = self._parse_datetime(self._first(row, "Date Read", "date_read"))
                date_added = self._parse_datetime(self._first(row, "Date Added", "date_added"))
                comparable_at = date_read or date_added
                if sync_at and comparable_at and comparable_at <= sync_at:
                    continue

                isbn = self._clean_isbn(self._first(row, "ISBN", "isbn"))
                isbn13 = self._clean_isbn(self._first(row, "ISBN13", "isbn13"))
                shelves = self._shelves(row)
                exclusive_shelf = self._first(row, "Exclusive Shelf", "exclusive_shelf")
                if exclusive_shelf and exclusive_shelf.lower() not in shelves:
                    shelves.insert(0, exclusive_shelf.lower())
                review = self._first(row, "My Review", "my_review", "review")
                rating = self._first(row, "My Rating", "my_rating", "rating")
                book_id = self._first(row, "Book Id", "book_id", "id")
                now = datetime.now(timezone.utc)

                metadata = {
                    "book_id": book_id,
                    "title": title,
                    "author": author,
                    "isbn": isbn,
                    "isbn13": isbn13,
                    "exclusive_shelf": exclusive_shelf,
                    "shelves": shelves,
                    "rating": self._int_or_none(rating),
                    "date_read": self._first(row, "Date Read", "date_read"),
                    "date_added": self._first(row, "Date Added", "date_added"),
                    "review": review,
                    "source_file": str(path),
                }
                books.append(
                    KnowledgeUnit(
                        source_project=SourceProject.GOODREADS_LIBRARY,
                        source_id=self._source_id(book_id, isbn13 or isbn, title, author),
                        source_entity_type="book",
                        title=self._format_title(title, author),
                        content=self._content(title, author, rating, shelves, review),
                        content_type=ContentType.ARTIFACT,
                        metadata=metadata,
                        tags=shelves,
                        created_at=date_added or date_read or now,
                        updated_at=date_read or date_added or now,
                    )
                )

        authors = self._author_units(books) if "author" in allowed_types else []
        shelves = self._shelf_units(books) if "shelf" in allowed_types else []
        if "author" in allowed_types:
            result.units.extend(authors)
        if "book" in allowed_types:
            result.units.extend(books)
        if "shelf" in allowed_types:
            result.units.extend(shelves)
        result.edges.extend(self._edges(books, authors, shelves, allowed_types))
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        paths: list[Path] = []
        for raw in re.split(r"[\n,]", self.path):
            text = raw.strip()
            if not text:
                continue
            path = Path(text).expanduser()
            if path.is_dir():
                paths.extend(sorted(child for child in path.rglob("*.csv") if child.is_file()))
            elif path.is_file():
                paths.append(path)
        return paths

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]

    def _source_id(self, book_id: str, isbn: str, title: str, author: str) -> str:
        if book_id:
            return f"goodreads_library:{book_id}"
        if isbn:
            return f"isbn:{isbn}"
        digest = hashlib.sha256(f"{title}|{author}".encode("utf-8")).hexdigest()[:24]
        return f"goodreads_library:{digest}"

    def _format_title(self, title: str, author: str) -> str:
        if title and author:
            return f"{title} by {author}"
        return title or author or "Untitled Goodreads book"

    def _content(self, title: str, author: str, rating: str, shelves: list[str], review: str) -> str:
        parts: list[str] = []
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

    def _shelves(self, row: dict[str, Any]) -> list[str]:
        shelves: list[str] = []
        for shelf in re.split(r",", self._first(row, "Bookshelves", "bookshelves", "shelves")):
            normalized = shelf.strip().lower()
            if normalized and normalized not in shelves:
                shelves.append(normalized)
        return shelves

    def _author_units(self, books: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for book in books:
            author = str(book.metadata.get("author") or "").strip()
            if author:
                grouped.setdefault(author, []).append(book)

        now = datetime.now(timezone.utc)
        units: list[KnowledgeUnit] = []
        for author, author_books in grouped.items():
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GOODREADS_LIBRARY,
                    source_id=self._author_source_id(author),
                    source_entity_type="author",
                    title=author,
                    content=f"Goodreads author: {author}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "author": author,
                        "book_count": len(author_books),
                        "source_file": sorted({str(book.metadata.get("source_file")) for book in author_books}),
                    },
                    tags=["author"],
                    created_at=min((book.created_at for book in author_books), default=now),
                    updated_at=max((book.updated_at for book in author_books), default=now),
                )
            )
        return units

    def _shelf_units(self, books: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for book in books:
            for shelf in book.metadata.get("shelves") or []:
                grouped.setdefault(str(shelf), []).append(book)

        now = datetime.now(timezone.utc)
        units: list[KnowledgeUnit] = []
        for shelf, shelf_books in grouped.items():
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GOODREADS_LIBRARY,
                    source_id=self._shelf_source_id(shelf),
                    source_entity_type="shelf",
                    title=f"Goodreads shelf: {shelf}",
                    content=f"Shelf: {shelf}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "shelf": shelf,
                        "book_count": len(shelf_books),
                        "source_file": sorted({str(book.metadata.get("source_file")) for book in shelf_books}),
                    },
                    tags=["shelf", shelf],
                    created_at=min((book.created_at for book in shelf_books), default=now),
                    updated_at=max((book.updated_at for book in shelf_books), default=now),
                )
            )
        return units

    def _edges(
        self,
        books: list[KnowledgeUnit],
        authors: list[KnowledgeUnit],
        shelves: list[KnowledgeUnit],
        allowed_types: set[str],
    ) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        author_ids = {str(author.metadata["author"]): author.source_id for author in authors}
        shelf_ids = {str(shelf.metadata["shelf"]): shelf.source_id for shelf in shelves}
        if {"book", "author"}.issubset(allowed_types):
            for book in books:
                author_id = author_ids.get(str(book.metadata.get("author") or ""))
                if author_id:
                    edges.append(self._edge(book.source_id, author_id, "book_author", EdgeRelation.RELATES_TO))
        if {"book", "shelf"}.issubset(allowed_types):
            for book in books:
                for shelf in book.metadata.get("shelves") or []:
                    shelf_id = shelf_ids.get(str(shelf))
                    if shelf_id:
                        edges.append(self._edge(shelf_id, book.source_id, "shelf_contains_book", EdgeRelation.CONTAINS))
        return self._dedupe_edges(edges)

    def _edge(self, from_id: str, to_id: str, relation_type: str, relation: EdgeRelation) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._edge_id(from_id, to_id, relation_type),
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=relation,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.GOODREADS_LIBRARY.value,
                "relation_type": relation_type,
            },
        )

    def _dedupe_edges(self, edges: list[KnowledgeEdge]) -> list[KnowledgeEdge]:
        by_id = {edge.id: edge for edge in edges}
        return list(by_id.values())

    def _author_source_id(self, author: str) -> str:
        digest = hashlib.sha256(author.strip().lower().encode("utf-8")).hexdigest()[:24]
        return f"goodreads_library:author:{digest}"

    def _shelf_source_id(self, shelf: str) -> str:
        digest = hashlib.sha256(shelf.strip().lower().encode("utf-8")).hexdigest()[:24]
        return f"goodreads_library:shelf:{digest}"

    def _edge_id(self, from_id: str, to_id: str, relation_type: str) -> str:
        digest = hashlib.sha256("|".join((from_id, to_id, relation_type)).encode("utf-8")).hexdigest()[:24]
        return f"goodreads-library-{relation_type}-{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _clean_isbn(self, value: str) -> str:
        return value.strip().strip('="').replace("-", "")

    def _int_or_none(self, value: str) -> int | None:
        try:
            return int(value)
        except ValueError:
            return None

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
