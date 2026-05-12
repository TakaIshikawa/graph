"""Adapter for StoryGraph reading history CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class StoryGraphReadingHistoryCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "storygraph_reading_history_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["book", "read"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and not set(entity_types).intersection(self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _iter_paths(self) -> list[Path]:
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if root.is_dir():
            return sorted(child for child in root.rglob("*.csv") if child.is_file())
        return []

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]

    def _unit_from_row(self, row: dict[str, Any], path: Path) -> KnowledgeUnit | None:
        title = self._first(row, "Title", "title", "Book Title", "book_title")
        if not title:
            return None

        authors = self._split_people(self._first(row, "Authors", "Author", "authors", "author"))
        isbn = self._clean_isbn(self._first(row, "ISBN", "isbn"))
        isbn13 = self._clean_isbn(self._first(row, "ISBN13", "ISBN/UID", "isbn13", "isbn_13"))
        date_read_text = self._first(row, "Date Read", "Last Date Read", "Read Date", "date_read", "date read")
        updated_text = self._first(row, "Updated At", "Last Updated", "Date Added", "updated_at", "date_added")
        date_read = self._parse_datetime(date_read_text)
        updated_at = self._parse_datetime(updated_text) or date_read
        created_at = date_read or updated_at or datetime.now(timezone.utc)
        rating = self._parse_float(self._first(row, "Star Rating", "Rating", "My Rating", "star_rating", "rating"))
        pages = self._parse_int(self._first(row, "Pages", "Number of Pages", "pages"))
        read_count = self._parse_int(self._first(row, "Read Count", "read_count", "Times Read"))
        moods = self._split_list(self._first(row, "Moods", "moods"))
        shelves = self._split_list(self._first(row, "Tags", "Shelves", "Bookshelves", "tags", "shelves"))
        review = self._first(row, "Review", "My Review", "review", "review_text")

        metadata = {
            "title": title,
            "authors": authors,
            "isbn": isbn,
            "isbn13": isbn13,
            "date_read": date_read.isoformat() if date_read else "",
            "read_count": read_count,
            "rating": rating,
            "moods": moods,
            "tags": shelves,
            "shelves": shelves,
            "pages": pages,
            "review": review,
            "source_file": str(path),
            "row": dict(row),
        }
        return KnowledgeUnit(
            source_project=SourceProject.STORYGRAPH_READING_HISTORY_CSV,
            source_id=self._source_id(title, authors, isbn13 or isbn, date_read_text, read_count, row),
            source_entity_type="read",
            title=self._format_title(title, authors),
            content=self._content(title, authors, date_read, rating, shelves, moods, review),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=self._dedupe(["storygraph", *shelves, *moods]),
            created_at=created_at,
            updated_at=updated_at or created_at,
        )

    def _source_id(
        self,
        title: str,
        authors: list[str],
        isbn: str,
        date_read: str,
        read_count: int | None,
        row: dict[str, Any],
    ) -> str:
        explicit = self._first(row, "ID", "Book ID", "Reading ID", "id", "book_id")
        if explicit:
            raw = explicit
        else:
            raw = "|".join([isbn, title, ";".join(authors), date_read, "" if read_count is None else str(read_count)])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"storygraph_reading_history_csv:{digest}"

    def _content(
        self,
        title: str,
        authors: list[str],
        date_read: datetime | None,
        rating: float | None,
        shelves: list[str],
        moods: list[str],
        review: str,
    ) -> str:
        parts = [f"Title: {title}"]
        if authors:
            parts.append(f"Authors: {', '.join(authors)}")
        if date_read:
            parts.append(f"Date read: {date_read.date().isoformat()}")
        if rating is not None:
            parts.append(f"Rating: {rating:g}/5")
        if shelves:
            parts.append(f"Tags: {', '.join(shelves)}")
        if moods:
            parts.append(f"Moods: {', '.join(moods)}")
        if review:
            parts.append(f"\nReview:\n{review}")
        return "\n".join(parts)

    def _format_title(self, title: str, authors: list[str]) -> str:
        return f"{title} by {', '.join(authors)}" if authors else title

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).lower(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.lower())
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _split_people(self, value: str) -> list[str]:
        return self._dedupe(part.strip() for part in re.split(r"\s*(?:,|;|\band\b)\s*", value) if part.strip())

    def _split_list(self, value: str) -> list[str]:
        return self._dedupe(part.strip().lower() for part in re.split(r"[,;|]", value) if part.strip())

    def _dedupe(self, values: Any) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _clean_isbn(self, value: str) -> str:
        return value.strip().strip('="').replace("-", "").replace(" ", "")

    def _parse_int(self, value: str) -> int | None:
        if not value:
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def _parse_float(self, value: str) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        text = value.strip()
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
