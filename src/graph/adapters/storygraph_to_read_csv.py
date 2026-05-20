"""Adapter for StoryGraph to-read CSV exports."""

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


class StoryGraphToReadCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "storygraph_to_read_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["reading_queue_item"]

    def __init__(self, path: str | Path | TextIO = "", file: TextIO | None = None) -> None:
        self.path = path
        self.file = file

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "reading_queue_item" not in set(entity_types or self.entity_types):
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
            return [(self._read_rows(self.file), self._source_name(self.file))]
        if hasattr(self.path, "read"):
            handle = self.path
            return [(self._read_rows(handle), self._source_name(handle))]
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
                    row_sets.append((self._read_rows(handle), path.name))
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
        return row_sets

    def _read_rows(self, handle: Any) -> list[dict[str, str]]:
        if hasattr(handle, "seek"):
            handle.seek(0)
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        return [{str(key).strip(): "" if value is None else str(value).strip() for key, value in row.items() if key is not None} for row in reader]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        if not any(self._text(value) for value in row.values()):
            return None

        title = self._first(row, "Title", "Book Title", "title")
        authors = self._split_people(self._first(row, "Authors", "Author", "authors", "author"))
        isbn = self._first(row, "ISBN", "ISBN13", "ISBN 13", "isbn", "isbn13")
        book_format = self._first(row, "Format", "Book Format", "format")
        pages = self._parse_int(self._first(row, "Pages", "Number of Pages", "Page Count", "pages"))
        date_added_text = self._first(row, "Date Added", "Added Date", "date_added", "added_at")
        date_added = self._parse_datetime(date_added_text)
        tags = self._split_tags(self._first(row, "Tags", "Shelves", "Bookshelves", "tags"))
        owned = self._parse_bool(self._first(row, "Owned", "owned", "Own", "I Own"))
        url = self._first(row, "URL", "Book URL", "Link", "url")

        if not title and not authors and not isbn and not url:
            return None

        now = datetime.now(timezone.utc)
        metadata = {
            "title": title,
            "authors": authors,
            "isbn": isbn,
            "format": book_format,
            "pages": pages,
            "tags": tags,
            "owned": owned,
            "date_added": date_added.isoformat() if date_added else date_added_text,
            "url": url,
            "source_file": source_file,
            "row": dict(row),
        }
        return KnowledgeUnit(
            source_project="storygraph_to_read_csv",
            source_id=self._source_id(title, authors, isbn, url),
            source_entity_type="reading_queue_item",
            title=self._format_title(title, authors),
            content=self._content(title, authors, isbn, book_format, pages, tags, owned, date_added, url),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=self._dedupe(["storygraph", "to-read", *tags]),
            created_at=date_added or now,
            updated_at=date_added or now,
        )

    def _content(
        self,
        title: str,
        authors: list[str],
        isbn: str,
        book_format: str,
        pages: int | None,
        tags: list[str],
        owned: bool | None,
        date_added: datetime | None,
        url: str,
    ) -> str:
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if authors:
            parts.append(f"Authors: {', '.join(authors)}")
        if isbn:
            parts.append(f"ISBN: {isbn}")
        if book_format:
            parts.append(f"Format: {book_format}")
        if pages is not None:
            parts.append(f"Pages: {pages}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if owned is not None:
            parts.append(f"Owned: {owned}")
        if date_added:
            parts.append(f"Date added: {date_added.date().isoformat()}")
        if url:
            parts.append(f"URL: {url}")
        return "\n".join(parts)

    def _format_title(self, title: str, authors: list[str]) -> str:
        if title and authors:
            return f"{title} by {', '.join(authors)}"
        return title or ", ".join(authors) or "Untitled StoryGraph to-read item"

    def _source_id(self, title: str, authors: list[str], isbn: str, url: str) -> str:
        raw = "|".join([self._stable_text(isbn or url), self._stable_text(title), ",".join(self._stable_text(author) for author in authors)])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"storygraph_to_read_csv:{digest}"

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

    def _split_people(self, value: str) -> list[str]:
        if not value:
            return []
        return self._dedupe(part.strip() for part in re.split(r"\s*(?:,|;|\band\b)\s*", value) if part.strip())

    def _split_tags(self, value: str) -> list[str]:
        if not value:
            return []
        return self._dedupe(self._stable_text(part) for part in re.split(r"[,;|]", value) if part.strip())

    def _parse_int(self, value: str) -> int | None:
        if not value:
            return None
        try:
            return int(float(value.replace(",", "")))
        except ValueError:
            return None

    def _parse_bool(self, value: str) -> bool | None:
        text = value.strip().casefold()
        if text in {"yes", "y", "true", "1", "owned"}:
            return True
        if text in {"no", "n", "false", "0", "not owned"}:
            return False
        return None

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
