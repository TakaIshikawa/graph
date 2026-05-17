"""Adapter for Goodreads reviews CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GoodreadsReviewsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "goodreads_reviews_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["book_review"]

    def __init__(self, path: str | Path | TextIO = "", file: TextIO | None = None) -> None:
        self.path = path
        self.file = file

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "book_review" not in set(entity_types or self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for rows, source_name in self._iter_row_sets():
            for row in rows:
                unit = self._unit_from_row(row, source_name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _iter_row_sets(self) -> list[tuple[list[dict[str, str]], str]]:
        if self.file is not None:
            return [(self._read_rows_from_handle(self.file), self._source_name(self.file))]
        if hasattr(self.path, "read"):
            handle = self.path
            return [(self._read_rows_from_handle(handle), self._source_name(handle))]
        if not self.path:
            return []

        root = Path(self.path).expanduser()
        paths: list[Path]
        if root.is_file() and root.suffix.lower() == ".csv":
            paths = [root]
        elif root.is_dir():
            paths = sorted(child for child in root.rglob("*.csv") if child.is_file())
        else:
            return []

        row_sets: list[tuple[list[dict[str, str]], str]] = []
        for path in paths:
            try:
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    row_sets.append((self._read_rows_from_handle(handle), path.name))
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
        return row_sets

    def _read_rows_from_handle(self, handle: Any) -> list[dict[str, str]]:
        if hasattr(handle, "seek"):
            handle.seek(0)
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        return [{str(key).strip(): "" if value is None else str(value).strip() for key, value in row.items() if key is not None} for row in reader]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        if not any(self._text(value) for value in row.values()):
            return None

        book_id = self._first(row, "Book Id", "Book ID", "book_id", "id")
        title = self._first(row, "Title", "title", "Book Title", "book_title")
        author = self._first(row, "Author", "author")
        author_lf = self._first(row, "Author l-f", "Author L-F", "author_lf")
        additional_authors = self._split_list(self._first(row, "Additional Authors", "additional_authors"))
        isbn = self._clean_isbn(self._first(row, "ISBN", "isbn"))
        isbn13 = self._clean_isbn(self._first(row, "ISBN13", "ISBN 13", "isbn13"))
        my_rating = self._parse_number(self._first(row, "My Rating", "my_rating", "rating"))
        average_rating = self._parse_number(self._first(row, "Average Rating", "average_rating"))
        publisher = self._first(row, "Publisher", "publisher")
        binding = self._first(row, "Binding", "binding")
        page_count = self._parse_int(self._first(row, "Number of Pages", "Pages", "number_of_pages", "pages"))
        year_published = self._parse_int(self._first(row, "Year Published", "year_published"))
        original_publication_year = self._parse_int(self._first(row, "Original Publication Year", "original_publication_year"))
        date_read_text = self._first(row, "Date Read", "date_read")
        date_added_text = self._first(row, "Date Added", "date_added")
        date_updated_text = self._first(row, "Date Updated", "date_updated", "Last Updated", "last_updated")
        date_read = self._parse_datetime(date_read_text)
        date_added = self._parse_datetime(date_added_text)
        date_updated = self._parse_datetime(date_updated_text)
        shelves = self._split_list(self._first(row, "Bookshelves", "Shelves", "Review Shelves", "bookshelves", "shelves"))
        exclusive_shelf = self._first(row, "Exclusive Shelf", "exclusive_shelf")
        review = self._first(row, "My Review", "Review", "my_review", "review")
        spoiler = self._parse_bool(self._first(row, "Spoiler", "spoiler"))
        private_notes = self._first(row, "Private Notes", "private_notes")
        read_count = self._parse_int(self._first(row, "Read Count", "read_count"))
        owned_copies = self._parse_int(self._first(row, "Owned Copies", "owned_copies"))

        if not title and not author and not book_id and not isbn and not isbn13:
            return None

        tag_values = self._dedupe([*shelves, exclusive_shelf])
        metadata = {
            "book_id": book_id,
            "title": title,
            "author": author,
            "author_lf": author_lf,
            "additional_authors": additional_authors,
            "isbn": isbn,
            "isbn13": isbn13,
            "my_rating": my_rating,
            "average_rating": average_rating,
            "publisher": publisher,
            "binding": binding,
            "page_count": page_count,
            "year_published": year_published,
            "original_publication_year": original_publication_year,
            "date_read": date_read.isoformat() if date_read else date_read_text,
            "date_added": date_added.isoformat() if date_added else date_added_text,
            "date_updated": date_updated.isoformat() if date_updated else date_updated_text,
            "shelves": shelves,
            "exclusive_shelf": exclusive_shelf,
            "review": review,
            "spoiler": spoiler,
            "private_notes": private_notes,
            "read_count": read_count,
            "owned_copies": owned_copies,
            "source_file": source_file,
            "row": dict(row),
        }
        now = datetime.now(timezone.utc)
        created_at = date_added or date_read or now
        updated_at = date_read or date_added or now
        return KnowledgeUnit(
            source_project="goodreads_reviews_csv",
            source_id=self._source_id(book_id, isbn13, isbn, title, author, author_lf, year_published, original_publication_year),
            source_entity_type="book_review",
            title=self._format_title(title, author or author_lf),
            content=self._content(title, author or author_lf, additional_authors, my_rating, average_rating, tag_values, review),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=self._dedupe(["goodreads", *tag_values]),
            created_at=created_at,
            updated_at=updated_at,
        )

    def _source_id(
        self,
        book_id: str,
        isbn13: str,
        isbn: str,
        title: str,
        author: str,
        author_lf: str,
        year_published: int | None,
        original_publication_year: int | None,
    ) -> str:
        if book_id:
            return f"goodreads_reviews_csv:{book_id}"
        raw = "|".join(
            [
                isbn13 or isbn,
                self._stable_text(title),
                self._stable_text(author or author_lf),
                "" if year_published is None else str(year_published),
                "" if original_publication_year is None else str(original_publication_year),
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"goodreads_reviews_csv:{digest}"

    def _content(
        self,
        title: str,
        author: str,
        additional_authors: list[str],
        my_rating: float | int | None,
        average_rating: float | int | None,
        tags: list[str],
        review: str,
    ) -> str:
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if author:
            parts.append(f"Author: {author}")
        if additional_authors:
            parts.append(f"Additional authors: {', '.join(additional_authors)}")
        if my_rating is not None:
            parts.append(f"My rating: {my_rating:g}/5")
        if average_rating is not None:
            parts.append(f"Average rating: {average_rating:g}/5")
        if tags:
            parts.append(f"Shelves: {', '.join(tags)}")
        if review:
            parts.append(f"\nReview:\n{review}")
        return "\n".join(parts)

    def _format_title(self, title: str, author: str) -> str:
        if title and author:
            return f"{title} by {author}"
        return title or author or "Untitled Goodreads review"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            text = self._text(value)
            if text:
                return text
        return ""

    def _split_list(self, value: str) -> list[str]:
        if not value:
            return []
        return self._dedupe(part.strip().lower() for part in re.split(r"[,;|]", value) if part.strip())

    def _clean_isbn(self, value: str) -> str:
        text = value.strip()
        if text.startswith('="') and text.endswith('"'):
            text = text[2:-1]
        return re.sub(r"[^0-9Xx]", "", text).upper()

    def _parse_number(self, value: str) -> float | int | None:
        if not value:
            return None
        try:
            number = float(value)
        except ValueError:
            return None
        return int(number) if number.is_integer() else number

    def _parse_int(self, value: str) -> int | None:
        number = self._parse_number(value)
        if isinstance(number, int):
            return number
        return None

    def _parse_bool(self, value: str) -> bool | None:
        text = value.strip().casefold()
        if text in {"true", "t", "yes", "y", "1"}:
            return True
        if text in {"false", "f", "no", "n", "0"}:
            return False
        return None

    def _parse_datetime(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            pass
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _stable_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().casefold())

    def _text(self, value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""
        return str(value).strip()

    def _dedupe(self, values: Any) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    def _source_name(self, handle: Any) -> str:
        name = self._text(getattr(handle, "name", ""))
        return Path(name).name if name else "<memory>"
