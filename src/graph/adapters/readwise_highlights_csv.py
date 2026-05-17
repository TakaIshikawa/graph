"""Adapter for Readwise highlights CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class ReadwiseHighlightsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "readwise_highlights_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["highlight"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "highlight" not in set(entity_types or self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path, source_file in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row_number, row in rows:
                unit = self._unit_from_row(row, source_file, row_number)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _iter_paths(self) -> list[tuple[Path, str]]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".csv":
            return [(root, root.name)]
        if root.is_dir():
            return [(child, child.relative_to(root).as_posix()) for child in sorted(root.rglob("*.csv")) if child.is_file()]
        return []

    def _read_rows(self, path: Path) -> list[tuple[int, dict[str, Any]]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            rows: list[tuple[int, dict[str, Any]]] = []
            for row_number, row in enumerate(reader, start=2):
                normalized = {str(key).strip(): value for key, value in row.items() if key is not None}
                if any(self._text(value) for value in normalized.values()):
                    rows.append((row_number, normalized))
            return rows

    def _unit_from_row(self, row: dict[str, Any], source_file: str, row_number: int) -> KnowledgeUnit | None:
        highlight = self._first(row, "Highlight", "Text", "Highlighted Text", "Quote")
        note = self._first(row, "Note", "Notes", "Annotation")
        if not highlight and not note:
            return None

        title = self._first(row, "Title", "Book Title", "Article Title", "Document Title", "Source Title")
        author = self._first(row, "Author", "Book Author", "Authors", "Source Author")
        source_type = self._first(row, "Source Type", "Category", "Type")
        url = self._first(row, "URL", "Source URL", "Book URL", "Article URL")
        location = self._first(row, "Location", "Location Type", "Position")
        tags = self._parse_tags(self._first(row, "Tags", "Tag"))
        highlighted_text = self._first(row, "Highlighted At", "Highlighted at", "Date", "Created At", "Created")
        updated_text = self._first(row, "Updated At", "Updated at", "Modified At", "Last Updated")
        highlighted_at = self._parse_datetime(highlighted_text)
        updated_at = self._parse_datetime(updated_text) or highlighted_at
        now = datetime.now(timezone.utc)

        metadata = {
            "source_file": source_file,
            "row_number": row_number,
            "highlight": highlight,
            "text": highlight,
            "title": title,
            "author": author,
            "source_type": source_type,
            "url": url,
            "location": location,
            "note": note,
            "tags": tags,
            "highlighted_at": highlighted_at.isoformat() if highlighted_at else highlighted_text,
            "updated_at": updated_at.isoformat() if updated_at else updated_text,
            "row": dict(row),
        }

        return KnowledgeUnit(
            source_project="readwise_highlights_csv",
            source_id=self._source_id(row, title, author, highlight, note, url, location, highlighted_text),
            source_entity_type="highlight",
            title=title or "Readwise highlight",
            content=self._content(highlight, note, title, author, source_type, url, location, tags),
            content_type=ContentType.INSIGHT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=tags,
            created_at=highlighted_at or updated_at or now,
            updated_at=updated_at or highlighted_at or now,
        )

    def _source_id(self, row: dict[str, Any], title: str, author: str, highlight: str, note: str, url: str, location: str, highlighted_at: str) -> str:
        explicit = self._first(row, "Highlight ID", "ID", "id", "highlight_id")
        if explicit:
            return f"readwise_highlights_csv:{explicit}"
        raw = "|".join([title, author, highlight, note, url, location, highlighted_at])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"readwise_highlights_csv:{digest}"

    def _content(self, highlight: str, note: str, title: str, author: str, source_type: str, url: str, location: str, tags: list[str]) -> str:
        parts = []
        if highlight:
            parts.append(highlight)
        if note:
            parts.append(f"Note: {note}")
        if title:
            parts.append(f"Title: {title}")
        if author:
            parts.append(f"Author: {author}")
        if source_type:
            parts.append(f"Source type: {source_type}")
        if url:
            parts.append(f"URL: {url}")
        if location:
            parts.append(f"Location: {location}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

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

    def _parse_tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for part in re.split(r"[,;|]", value):
            tag = part.strip().lstrip("#").strip()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _parse_datetime(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y"):
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

    def _text(self, value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""
        return str(value).strip()
