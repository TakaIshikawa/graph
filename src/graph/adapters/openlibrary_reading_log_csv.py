"""Adapter for OpenLibrary reading log CSV exports."""

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


class OpenLibraryReadingLogCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "openlibrary_reading_log_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["book"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "book" not in set(entity_types or self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.csv") if child.is_file())

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = self._first(row, "Title", "Book", "Name")
        author = self._first(row, "Author", "Authors")
        shelf = self._first(row, "Shelf", "Status", "Bookshelf", "Read Status")
        isbn = self._first(row, "ISBN", "ISBN10", "ISBN13", "ISBN 13")
        work_id = self._openlibrary_id(self._first(row, "Work ID", "OpenLibrary Work ID", "OL Work ID", "Work Key"))
        edition_id = self._openlibrary_id(self._first(row, "Edition ID", "OpenLibrary Edition ID", "OL Edition ID", "Edition Key"))
        started_at = self._parse_datetime(self._first(row, "Date Started", "Started", "Started At", "Start Date"))
        finished_at = self._parse_datetime(self._first(row, "Date Finished", "Finished", "Finished At", "Finish Date", "Date Read"))
        updated_at = self._parse_datetime(self._first(row, "Updated", "Updated At", "Last Modified")) or finished_at or started_at
        rating = self._parse_number(self._first(row, "Rating", "My Rating", "Stars"))
        notes = self._first(row, "Notes", "Review", "Comment", "Comments")
        if not title and not isbn and not work_id and not edition_id:
            return None
        metadata = {
            "title": title,
            "author": author,
            "shelf": shelf,
            "isbn": isbn,
            "work_id": work_id,
            "edition_id": edition_id,
            "started_at": started_at.isoformat() if started_at else self._first(row, "Date Started", "Started", "Started At", "Start Date"),
            "finished_at": finished_at.isoformat() if finished_at else self._first(row, "Date Finished", "Finished", "Finished At", "Finish Date", "Date Read"),
            "updated_at": updated_at.isoformat() if updated_at else self._first(row, "Updated", "Updated At", "Last Modified"),
            "rating": rating,
            "notes": notes,
            "source_file": source_file,
            "row": dict(row),
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.OPENLIBRARY_READING_LOG_CSV,
            source_id=self._source_id(work_id, edition_id, isbn, title, author),
            source_entity_type="book",
            title=title or isbn or work_id or edition_id,
            content=self._content(metadata),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(item for item in ["openlibrary", "book", shelf] if item)),
            created_at=started_at or finished_at or updated_at or now,
            updated_at=updated_at or now,
        )

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = [metadata.get("title")]
        for key, label in (("author", "Author"), ("shelf", "Shelf"), ("isbn", "ISBN"), ("work_id", "Work ID"), ("edition_id", "Edition ID"), ("started_at", "Started"), ("finished_at", "Finished"), ("rating", "Rating"), ("notes", "Notes")):
            if metadata.get(key) not in ("", None):
                parts.append(f"{label}: {metadata[key]}")
        return "\n".join(str(item) for item in parts if item)

    def _source_id(self, work_id: str, edition_id: str, isbn: str, title: str, author: str) -> str:
        stable = work_id or edition_id or re.sub(r"[^0-9Xx]", "", isbn).casefold()
        raw = stable or "|".join([self._normalized(title), self._normalized(author)])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"openlibrary_reading_log_csv:{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        compact = {self._normalize_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = compact.get(self._normalize_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _openlibrary_id(self, value: str) -> str:
        text = value.strip()
        if text.startswith("/works/") or text.startswith("/books/"):
            return text.rsplit("/", 1)[-1]
        return text

    def _parse_number(self, value: str) -> float | int | None:
        if not value:
            return None
        try:
            number = float(value.strip())
            return int(number) if number.is_integer() else number
        except ValueError:
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
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    def _normalized(self, value: str) -> str:
        return " ".join(value.casefold().split())

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
