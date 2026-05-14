"""Adapter for Google Play Books notes and highlights CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GooglePlayBooksNotesCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_play_books_notes_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["book_note"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else set(self.entity_types)
        if "book_note" not in allowed:
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.updated_at, unit.source_id)))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        book_title = first(row, "Book Title", "Title", "book_title")
        author = first(row, "Author", "Authors", "author")
        highlight = first(row, "Highlight", "Highlighted Text", "Selection", "Text", "Quote")
        note = first(row, "Note", "Notes", "Annotation", "Comment")
        location = first(row, "Location", "Page", "Position", "Chapter")
        if not (highlight or note):
            return None

        created_at = parse_datetime(first(row, "Created At", "Date Created", "Created", "Timestamp", "Date"))
        event_at = created_at or datetime.now(timezone.utc)
        color = first(row, "Color", "Highlight Color", "Colour")
        metadata = clean_metadata(
            {
                "book_title": book_title,
                "author": author,
                "highlight": highlight,
                "note": note,
                "color": color,
                "location": location,
                "created_at": created_at.isoformat() if created_at else None,
                "source_file": source_file,
                "row": dict(row),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_PLAY_BOOKS_NOTES_CSV,
            source_id=digest_source_id("google_play_books_notes_csv", book_title, highlight, note, location),
            source_entity_type="book_note",
            title=self._title(book_title, highlight, note),
            content=self._content(book_title, author, highlight, note, color, location),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["google_play_books", "book_note"],
            created_at=event_at,
            updated_at=event_at,
        )

    def _title(self, book_title: str, highlight: str, note: str) -> str:
        if book_title:
            return f"Google Play Books note: {book_title}"
        text = highlight or note
        return f"Google Play Books note: {text[:80]}"

    def _content(self, book_title: str, author: str, highlight: str, note: str, color: str, location: str) -> str:
        parts: list[str] = []
        if book_title:
            parts.append(f"Book: {book_title}")
        if author:
            parts.append(f"Author: {author}")
        if location:
            parts.append(f"Location: {location}")
        if color:
            parts.append(f"Color: {color}")
        if highlight:
            parts.append(f"\nHighlight:\n{highlight}")
        if note:
            parts.append(f"\nNote:\n{note}")
        return "\n".join(parts)
