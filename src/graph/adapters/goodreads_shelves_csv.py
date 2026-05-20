"""Adapter for Goodreads shelf membership CSV exports."""

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


class GoodreadsShelvesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "goodreads_shelves_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["book_shelf_item"]

    def __init__(self, path: str | Path | TextIO = "", file: TextIO | None = None) -> None:
        self.path = path
        self.file = file

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "book_shelf_item" not in set(entity_types or self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for rows, source_file in self._iter_row_sets():
            for row in rows:
                unit = self._unit_from_row(row, source_file)
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
        title = self._first(row, "Title", "Book Title", "title", "book_title")
        author = self._first(row, "Author", "Authors", "author", "authors")
        exclusive_shelf = self._normalize_shelf(self._first(row, "Exclusive Shelf", "exclusive_shelf", "Shelf"))
        shelves = self._split_shelves(self._first(row, "Bookshelves", "Shelves", "bookshelves", "shelves"))
        rating = self._parse_number(self._first(row, "My Rating", "Rating", "my_rating", "rating"))
        date_added_text = self._first(row, "Date Added", "date_added", "Added Date")
        date_read_text = self._first(row, "Date Read", "date_read", "Read Date")
        date_added = self._parse_datetime(date_added_text)
        date_read = self._parse_datetime(date_read_text)
        url = self._first(row, "URL", "Book URL", "Link", "url", "book_url")

        if not title and not author and not book_id and not url and not exclusive_shelf and not shelves:
            return None

        all_shelves = self._dedupe([exclusive_shelf, *shelves])
        best_date = self._best_datetime(date_added, date_read)
        metadata = {
            "book_id": book_id,
            "title": title,
            "author": author,
            "exclusive_shelf": exclusive_shelf,
            "shelves": shelves,
            "all_shelves": all_shelves,
            "rating": rating,
            "my_rating": rating,
            "date_added": date_added.isoformat() if date_added else date_added_text,
            "date_read": date_read.isoformat() if date_read else date_read_text,
            "url": url,
            "source_file": source_file,
            "row": dict(row),
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project="goodreads_shelves_csv",
            source_id=self._source_id(book_id, title, author, exclusive_shelf, shelves, url),
            source_entity_type="book_shelf_item",
            title=self._format_title(title, author, all_shelves),
            content=self._content(title, author, rating, date_added, date_read, url, all_shelves),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=self._dedupe(["goodreads", "book", *all_shelves]),
            created_at=date_added or date_read or now,
            updated_at=best_date or now,
        )

    def _source_id(self, book_id: str, title: str, author: str, exclusive_shelf: str, shelves: list[str], url: str) -> str:
        raw = "|".join(
            [
                self._stable_text(book_id or url),
                self._stable_text(title),
                self._stable_text(author),
                self._stable_text(exclusive_shelf),
                ",".join(self._stable_text(shelf) for shelf in shelves),
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"goodreads_shelves_csv:{digest}"

    def _content(
        self,
        title: str,
        author: str,
        rating: float | int | None,
        date_added: datetime | None,
        date_read: datetime | None,
        url: str,
        shelves: list[str],
    ) -> str:
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if author:
            parts.append(f"Author: {author}")
        if shelves:
            parts.append(f"Shelves: {', '.join(shelves)}")
        if rating is not None:
            parts.append(f"My rating: {rating:g}/5")
        if date_added:
            parts.append(f"Date added: {date_added.date().isoformat()}")
        if date_read:
            parts.append(f"Date read: {date_read.date().isoformat()}")
        if url:
            parts.append(f"URL: {url}")
        return "\n".join(parts)

    def _format_title(self, title: str, author: str, shelves: list[str]) -> str:
        base = title or author or "Untitled Goodreads shelf item"
        if title and author:
            base = f"{title} by {author}"
        if shelves:
            return f"{base} [{', '.join(shelves)}]"
        return base

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

    def _split_shelves(self, value: str) -> list[str]:
        if not value:
            return []
        return self._dedupe(self._normalize_shelf(part) for part in re.split(r"[,;]", value) if part.strip())

    def _normalize_shelf(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().casefold())

    def _parse_number(self, value: str) -> float | int | None:
        if not value:
            return None
        try:
            number = float(value)
        except ValueError:
            return None
        return int(number) if number.is_integer() else number

    def _parse_datetime(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        for candidate in (text, text.replace("Z", "+00:00")):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate))
            except ValueError:
                pass
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _best_datetime(self, *values: datetime | None) -> datetime | None:
        dates = [value for value in values if value is not None]
        return max(dates) if dates else None

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
