"""Adapter for Kobo annotations CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, first, iter_paths, parse_datetime, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class KoboAnnotationsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "kobo_annotations_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["annotation"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "annotation" not in entity_types:
            return result

        sync_at = self._sync_at(since)
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        highlight = first(row, "Annotation", "Highlight", "Highlighted Text", "Text", "Quote")
        note = first(row, "Note", "Notes", "Annotation Note", "Note Text")
        if not highlight and not note:
            return None

        book_title = first(row, "Book Title", "Title", "Book")
        author = first(row, "Author", "Authors")
        chapter = first(row, "Chapter", "Chapter Title", "Section")
        page = first(row, "Page", "Page Number")
        location = first(row, "Location", "Position", "Chapter Progress") or page
        color = first(row, "Color", "Colour")
        created_text = first(row, "Created Date", "Date Created", "Created", "Created At")
        modified_text = first(row, "Modified Date", "Date Modified", "Modified", "Updated At")
        created_at = parse_datetime(created_text)
        modified_at = parse_datetime(modified_text)
        now = datetime.now(timezone.utc)

        return KnowledgeUnit(
            source_project="kobo_annotations_csv",
            source_id=digest_source_id("kobo_annotations_csv", book_title, author, chapter, location, page, highlight, note, created_text, index),
            source_entity_type="annotation",
            title=self._title(book_title, highlight or note),
            content=self._content(book_title, author, chapter, location, page, color, highlight, note),
            content_type=ContentType.INSIGHT,
            metadata=clean_metadata(
                {
                    "book_title": book_title,
                    "author": author,
                    "highlight": highlight,
                    "annotation": highlight,
                    "note": note,
                    "chapter": chapter,
                    "location": location,
                    "page": page,
                    "color": color,
                    "created_date": created_at.isoformat() if created_at else created_text,
                    "modified_date": modified_at.isoformat() if modified_at else modified_text,
                    "source_file": source_file,
                    "row": dict(row),
                }
            ),
            tags=self._tags(color),
            created_at=created_at or modified_at or now,
            updated_at=modified_at or created_at or now,
        )

    def _title(self, book_title: str, text: str) -> str:
        if book_title:
            return f"Kobo annotation: {book_title}"
        return f"Kobo annotation: {text[:80]}" if text else "Kobo annotation"

    def _content(self, book_title: str, author: str, chapter: str, location: str, page: str, color: str, highlight: str, note: str) -> str:
        parts: list[str] = []
        if book_title:
            parts.append(f"Book: {book_title}")
        if author:
            parts.append(f"Author: {author}")
        if chapter:
            parts.append(f"Chapter: {chapter}")
        if page and page != location:
            parts.append(f"Page: {page}")
        if location:
            parts.append(f"Location: {location}")
        if color:
            parts.append(f"Color: {color}")
        if highlight:
            parts.append(f"Highlight: {highlight}")
        if note:
            parts.append(f"Note: {note}")
        return "\n".join(parts)

    def _tags(self, color: str) -> list[str]:
        tags = ["kobo", "annotation"]
        if color:
            tags.append(color.strip().casefold())
        return tags

    def _sync_at(self, since: SyncState | None) -> datetime | None:
        if since is None:
            return None
        value = since.last_sync_at
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
