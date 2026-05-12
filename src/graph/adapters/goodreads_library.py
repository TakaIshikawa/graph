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
        return ["author", "book", "copy", "publisher", "review", "series", "shelf"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types) if entity_types is not None else {"author", "book", "copy", "publisher", "series", "shelf"}
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = self._sync_datetime(since) if since else None
        books: list[KnowledgeUnit] = []
        copies: list[KnowledgeUnit] = []
        reviews: list[KnowledgeUnit] = []
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
                series = self._series_metadata(row, title)
                publisher = self._first(row, "Publisher", "publisher", "Original Publisher", "original_publisher")
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
                    "publisher": publisher,
                    "source_file": str(path),
                }
                if series:
                    metadata["series"] = series
                book = KnowledgeUnit(
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
                books.append(book)
                if review:
                    reviews.append(self._review_unit(book, row, review, rating, shelves, path, date_read or date_added or now))
                copy_metadata = self._copy_metadata(row)
                if copy_metadata:
                    copies.append(self._copy_unit(book, copy_metadata, row, path, date_added or date_read or now))

        authors = self._author_units(books) if "author" in allowed_types else []
        publishers = self._publisher_units(books) if "publisher" in allowed_types else []
        series = self._series_units(books) if "series" in allowed_types else []
        shelves = self._shelf_units(books) if "shelf" in allowed_types else []
        if "author" in allowed_types:
            result.units.extend(authors)
        if "book" in allowed_types:
            result.units.extend(books)
        if "copy" in allowed_types:
            result.units.extend(copies)
        if "publisher" in allowed_types:
            result.units.extend(publishers)
        if "review" in allowed_types:
            result.units.extend(reviews)
        if "series" in allowed_types:
            result.units.extend(series)
        if "shelf" in allowed_types:
            result.units.extend(shelves)
        result.edges.extend(self._edges(books, copies, reviews, authors, publishers, series, shelves, allowed_types))
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

    def _copy_metadata(self, row: dict[str, Any]) -> dict[str, str]:
        fields = {
            "condition": self._first(row, "Condition", "Owned Copy Condition", "Book Condition", "condition"),
            "date_acquired": self._first(row, "Date Acquired", "Acquired Date", "Purchase Date", "date_acquired"),
            "purchase_location": self._first(row, "Purchase Location", "Purchased From", "Store", "purchase_location"),
            "format": self._first(row, "Format", "Binding", "Book Format", "format"),
            "owned_copy_id": self._first(row, "Owned Copy Id", "Owned Copy ID", "Copy Id", "copy_id"),
        }
        return {key: value for key, value in fields.items() if value}

    def _copy_unit(
        self,
        book: KnowledgeUnit,
        copy_metadata: dict[str, str],
        row: dict[str, Any],
        path: Path,
        fallback_at: datetime,
    ) -> KnowledgeUnit:
        acquired = self._parse_datetime(copy_metadata.get("date_acquired", ""))
        metadata = {
            **copy_metadata,
            "book_source_id": book.source_id,
            "book_id": book.metadata.get("book_id"),
            "title": book.metadata.get("title"),
            "author": book.metadata.get("author"),
            "source_file": str(path),
        }
        return KnowledgeUnit(
            source_project=SourceProject.GOODREADS_LIBRARY,
            source_id=self._copy_source_id(book.source_id, copy_metadata, row),
            source_entity_type="copy",
            title=f"Owned copy of {book.title}",
            content=self._copy_content(book, copy_metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["owned-copy"],
            created_at=acquired or fallback_at,
            updated_at=acquired or fallback_at,
        )

    def _review_unit(
        self,
        book: KnowledgeUnit,
        row: dict[str, Any],
        review: str,
        rating: str,
        shelves: list[str],
        path: Path,
        fallback_at: datetime,
    ) -> KnowledgeUnit:
        metadata = {
            "book_source_id": book.source_id,
            "book_id": book.metadata.get("book_id"),
            "title": book.metadata.get("title"),
            "author": book.metadata.get("author"),
            "isbn": book.metadata.get("isbn"),
            "isbn13": book.metadata.get("isbn13"),
            "review": review,
            "rating": self._int_or_none(rating),
            "date_read": self._first(row, "Date Read", "date_read"),
            "date_added": self._first(row, "Date Added", "date_added"),
            "shelves": shelves,
            "source_file": str(path),
        }
        return KnowledgeUnit(
            source_project=SourceProject.GOODREADS_LIBRARY,
            source_id=self._review_source_id(book.source_id, review),
            source_entity_type="review",
            title=f"Goodreads review: {book.title}",
            content=review,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["review", *shelves],
            created_at=fallback_at,
            updated_at=fallback_at,
        )

    def _copy_content(self, book: KnowledgeUnit, copy_metadata: dict[str, str]) -> str:
        parts = [f"Book: {book.title}"]
        labels = {
            "condition": "Condition",
            "date_acquired": "Date acquired",
            "purchase_location": "Purchase location",
            "format": "Format",
        }
        for key, label in labels.items():
            if copy_metadata.get(key):
                parts.append(f"{label}: {copy_metadata[key]}")
        return "\n".join(parts)

    def _shelves(self, row: dict[str, Any]) -> list[str]:
        shelves: list[str] = []
        for shelf in re.split(r",", self._first(row, "Bookshelves", "bookshelves", "shelves")):
            normalized = shelf.strip().lower()
            if normalized and normalized not in shelves:
                shelves.append(normalized)
        return shelves

    def _series_metadata(self, row: dict[str, Any], title: str) -> dict[str, Any]:
        explicit = self._first(row, "Series", "Series Name", "Book Series", "series", "series_name", "book_series")
        sequence_text = self._first(row, "Series Number", "Book Number", "Series Position", "Number in Series")
        parsed_name = ""
        parsed_sequence = ""
        match = re.search(r"\((?P<series>[^()]*?)(?:,\s*#(?P<sequence>[\w. -]+))\)\s*$", title)
        if match:
            parsed_name = match.group("series").strip()
            parsed_sequence = (match.group("sequence") or "").strip()

        name = explicit or parsed_name
        if not name:
            return {}
        sequence = sequence_text or parsed_sequence
        metadata: dict[str, Any] = {
            "name": name,
            "source": "column" if explicit else "title",
        }
        if sequence:
            metadata["sequence"] = self._number_or_text(sequence)
        return metadata

    def _number_or_text(self, value: str) -> int | float | str:
        text = value.strip().lstrip("#").strip()
        try:
            number = float(text)
        except ValueError:
            return text
        return int(number) if number.is_integer() else number

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
            ratings = [rating for book in shelf_books if (rating := book.metadata.get("rating")) is not None]
            book_source_ids = sorted(book.source_id for book in shelf_books)
            authors = sorted({str(book.metadata.get("author") or "") for book in shelf_books if book.metadata.get("author")})
            date_added = [
                parsed
                for book in shelf_books
                if (parsed := self._parse_datetime(str(book.metadata.get("date_added") or ""))) is not None
            ]
            date_read = [
                parsed
                for book in shelf_books
                if (parsed := self._parse_datetime(str(book.metadata.get("date_read") or ""))) is not None
            ]
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
                        "book_source_ids": book_source_ids,
                        "authors": authors,
                        "rating_count": len(ratings),
                        "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
                        "read_count": self._shelf_status_count(shelf_books, "read"),
                        "to_read_count": self._shelf_status_count(shelf_books, "to-read"),
                        "currently_reading_count": self._shelf_status_count(shelf_books, "currently-reading"),
                        "reviewed_count": sum(1 for book in shelf_books if book.metadata.get("review")),
                        "first_read_at": min(date_read).isoformat() if date_read else None,
                        "latest_read_at": max(date_read).isoformat() if date_read else None,
                        "first_date_added": min(date_added).isoformat() if date_added else "",
                        "latest_date_read": max(date_read).isoformat() if date_read else "",
                        "top_authors": self._top_authors(shelf_books),
                        "source_file": sorted({str(book.metadata.get("source_file")) for book in shelf_books}),
                    },
                    tags=["shelf", shelf],
                    created_at=min((book.created_at for book in shelf_books), default=now),
                    updated_at=max((book.updated_at for book in shelf_books), default=now),
                )
            )
        return units

    def _shelf_status_count(self, books: list[KnowledgeUnit], status: str) -> int:
        return sum(1 for book in books if str(book.metadata.get("exclusive_shelf") or "").strip().lower() == status)

    def _top_authors(self, books: list[KnowledgeUnit]) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for book in books:
            author = str(book.metadata.get("author") or "").strip()
            if author:
                counts[author] = counts.get(author, 0) + 1
        return [{"author": author, "book_count": count} for author, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]]

    def _publisher_units(self, books: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for book in books:
            publisher = str(book.metadata.get("publisher") or "").strip()
            if not publisher:
                continue
            key = publisher.casefold()
            names.setdefault(key, publisher)
            grouped.setdefault(key, []).append(book)

        now = datetime.now(timezone.utc)
        units: list[KnowledgeUnit] = []
        for key, publisher_books in grouped.items():
            publisher = names[key]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GOODREADS_LIBRARY,
                    source_id=self._publisher_source_id(publisher),
                    source_entity_type="publisher",
                    title=publisher,
                    content=f"Goodreads publisher: {publisher}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "publisher": publisher,
                        "book_count": len(publisher_books),
                        "book_source_ids": [book.source_id for book in publisher_books],
                        "source_file": sorted({str(book.metadata.get("source_file")) for book in publisher_books}),
                    },
                    tags=["publisher"],
                    created_at=min((book.created_at for book in publisher_books), default=now),
                    updated_at=max((book.updated_at for book in publisher_books), default=now),
                )
            )
        return units

    def _series_units(self, books: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        names: dict[str, str] = {}
        for book in books:
            series = book.metadata.get("series") or {}
            name = str(series.get("name") or "").strip()
            if not name:
                continue
            key = name.casefold()
            names.setdefault(key, name)
            grouped.setdefault(key, []).append(book)

        now = datetime.now(timezone.utc)
        units: list[KnowledgeUnit] = []
        for key, series_books in grouped.items():
            name = names[key]
            ordered_books = sorted(series_books, key=self._series_book_sort_key)
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.GOODREADS_LIBRARY,
                    source_id=self._series_source_id(name),
                    source_entity_type="series",
                    title=f"Goodreads series: {name}",
                    content=f"Series: {name}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "series": name,
                        "book_count": len(series_books),
                        "books": [
                            {
                                "source_id": book.source_id,
                                "title": book.metadata.get("title"),
                                "author": book.metadata.get("author"),
                                "sequence": (book.metadata.get("series") or {}).get("sequence"),
                            }
                            for book in ordered_books
                        ],
                        "source_file": sorted({str(book.metadata.get("source_file")) for book in series_books}),
                    },
                    tags=["series"],
                    created_at=min((book.created_at for book in series_books), default=now),
                    updated_at=max((book.updated_at for book in series_books), default=now),
                )
            )
        return units

    def _series_book_sort_key(self, book: KnowledgeUnit) -> tuple[int, float | str, datetime, str]:
        sequence = (book.metadata.get("series") or {}).get("sequence")
        if isinstance(sequence, int | float):
            return (0, sequence, book.created_at, book.source_id)
        if sequence:
            return (1, str(sequence), book.created_at, book.source_id)
        return (2, "", book.created_at, book.source_id)

    def _edges(
        self,
        books: list[KnowledgeUnit],
        copies: list[KnowledgeUnit],
        reviews: list[KnowledgeUnit],
        authors: list[KnowledgeUnit],
        publishers: list[KnowledgeUnit],
        series: list[KnowledgeUnit],
        shelves: list[KnowledgeUnit],
        allowed_types: set[str],
    ) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        author_ids = {str(author.metadata["author"]): author.source_id for author in authors}
        publisher_ids = {str(publisher.metadata["publisher"]).casefold(): publisher.source_id for publisher in publishers}
        series_ids = {str(unit.metadata["series"]).casefold(): unit.source_id for unit in series}
        shelf_ids = {str(shelf.metadata["shelf"]): shelf.source_id for shelf in shelves}
        if {"book", "copy"}.issubset(allowed_types):
            book_ids = {book.source_id for book in books}
            for copy in copies:
                book_id = str(copy.metadata.get("book_source_id") or "")
                if book_id in book_ids:
                    edges.append(self._edge(book_id, copy.source_id, "book_contains_copy", EdgeRelation.CONTAINS))
        if {"book", "review"}.issubset(allowed_types):
            book_ids = {book.source_id for book in books}
            for review in reviews:
                book_id = str(review.metadata.get("book_source_id") or "")
                if book_id in book_ids:
                    edges.append(self._edge(book_id, review.source_id, "book_contains_review", EdgeRelation.CONTAINS))
        if {"book", "author"}.issubset(allowed_types):
            for book in books:
                author_id = author_ids.get(str(book.metadata.get("author") or ""))
                if author_id:
                    edges.append(self._edge(book.source_id, author_id, "book_author", EdgeRelation.RELATES_TO))
        if {"book", "publisher"}.issubset(allowed_types):
            for book in books:
                publisher_name = str(book.metadata.get("publisher") or "").strip()
                publisher_id = publisher_ids.get(publisher_name.casefold())
                if publisher_id:
                    edges.append(self._edge(book.source_id, publisher_id, "book_publisher", EdgeRelation.RELATES_TO))
        if {"book", "shelf"}.issubset(allowed_types):
            for book in books:
                for shelf in book.metadata.get("shelves") or []:
                    shelf_id = shelf_ids.get(str(shelf))
                    if shelf_id:
                        edges.append(self._edge(shelf_id, book.source_id, "shelf_contains_book", EdgeRelation.CONTAINS))
        if {"book", "series"}.issubset(allowed_types):
            for book in books:
                series_name = str((book.metadata.get("series") or {}).get("name") or "").strip()
                series_id = series_ids.get(series_name.casefold())
                if series_id:
                    edges.append(self._edge(series_id, book.source_id, "series_contains_book", EdgeRelation.CONTAINS))
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

    def _series_source_id(self, series: str) -> str:
        digest = hashlib.sha256(series.strip().lower().encode("utf-8")).hexdigest()[:24]
        return f"goodreads_library:series:{digest}"

    def _publisher_source_id(self, publisher: str) -> str:
        digest = hashlib.sha256(publisher.strip().casefold().encode("utf-8")).hexdigest()[:24]
        return f"goodreads_library:publisher:{digest}"

    def _copy_source_id(self, book_source_id: str, copy_metadata: dict[str, str], row: dict[str, Any]) -> str:
        explicit = copy_metadata.get("owned_copy_id") or self._first(row, "Owned Copy ID", "Copy ID")
        raw = explicit or "|".join(
            [
                book_source_id,
                copy_metadata.get("condition", ""),
                copy_metadata.get("date_acquired", ""),
                copy_metadata.get("purchase_location", ""),
                copy_metadata.get("format", ""),
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"goodreads_library:copy:{digest}"

    def _review_source_id(self, book_source_id: str, review: str) -> str:
        digest = hashlib.sha256(f"{book_source_id}|{review}".encode("utf-8")).hexdigest()[:24]
        return f"goodreads_library:review:{digest}"

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
