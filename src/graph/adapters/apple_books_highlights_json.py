"""Adapter for Apple Books highlight and note JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class AppleBooksHighlightsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "apple_books_highlights_json"

    @property
    def entity_types(self) -> list[str]:
        return ["book", "highlight", "note"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else {"highlight", "note"}
        if not allowed.intersection(self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit_from_record(record, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        annotations = sorted(result.units, key=lambda unit: (unit.updated_at, unit.source_id))
        books = self._book_units(annotations) if "book" in allowed else []
        result.units = []
        if "book" in allowed:
            result.units.extend(books)
        for entity_type in ("highlight", "note"):
            if entity_type in allowed:
                result.units.extend(unit for unit in annotations if unit.source_entity_type == entity_type)
        if "book" in allowed and {"highlight", "note"}.intersection(allowed):
            result.edges.extend(self._book_annotation_edges(books, annotations, allowed))
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, dict):
            for key in ("highlights", "notes", "annotations", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    parsed = value
                    break
        return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []

    def _unit_from_record(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        selected_text = first(record, "selectedText", "selected_text", "highlight", "text", "quote")
        note_text = first(record, "note", "noteText", "note_text", "comment")
        if not selected_text and not note_text:
            return None

        explicit_type = first(record, "type", "kind", "annotationType", "annotation_type").casefold()
        entity_type = "note" if explicit_type == "note" or (note_text and not selected_text) else "highlight"
        created = parse_datetime(first(record, "created", "createdAt", "creationDate", "dateCreated", "created_at"))
        modified = parse_datetime(first(record, "modified", "modifiedAt", "updated", "updatedAt", "dateModified", "modified_at"))
        updated_at = modified or created or datetime.now(timezone.utc)
        book_title = first(record, "bookTitle", "book_title", "title", "book")
        author = first(record, "author", "authors", "bookAuthor", "book_author")
        location = first(record, "location", "cfi", "page", "chapter")
        identifier = first(record, "id", "uuid", "annotationId", "annotation_id", "assetId", "asset_id", "isbn")

        metadata = clean_metadata(
            {
                "book_title": book_title,
                "author": author,
                "selected_text": selected_text,
                "note": note_text,
                "location": location,
                "page": first(record, "page", "pageNumber", "page_number"),
                "chapter": first(record, "chapter"),
                "identifier": identifier,
                "created_at": created.isoformat() if created else "",
                "modified_at": modified.isoformat() if modified else "",
                "source_file": source_file,
                "record": dict(record),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.APPLE_BOOKS_HIGHLIGHTS_JSON,
            source_id=self._source_id(entity_type, identifier, book_title, location, selected_text, note_text, created, index),
            source_entity_type=entity_type,
            title=self._title(entity_type, book_title, selected_text or note_text),
            content=self._content(book_title, author, location, selected_text, note_text),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["apple_books", entity_type],
            created_at=created or updated_at,
            updated_at=updated_at,
        )

    def _source_id(
        self,
        entity_type: str,
        identifier: str,
        book_title: str,
        location: str,
        selected_text: str,
        note_text: str,
        created: datetime | None,
        index: int,
    ) -> str:
        stable = identifier or "|".join([book_title, location, selected_text, note_text, created.isoformat() if created else str(index)])
        return digest_source_id(f"apple_books_highlights_json:{entity_type}", stable)

    def _book_units(self, annotations: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[tuple[str, str, str], list[KnowledgeUnit]] = {}
        for annotation in annotations:
            key = self._book_identity(annotation.metadata)
            if any(key):
                grouped.setdefault(key, []).append(annotation)

        units: list[KnowledgeUnit] = []
        for identity, book_annotations in grouped.items():
            first_unit = book_annotations[0]
            title = str(first_unit.metadata.get("book_title") or identity[2] or "Apple Books book")
            author = str(first_unit.metadata.get("author") or "")
            created_at = min(unit.created_at for unit in book_annotations)
            updated_at = max(unit.updated_at for unit in book_annotations)
            highlights = [unit for unit in book_annotations if unit.source_entity_type == "highlight"]
            notes = [unit for unit in book_annotations if unit.source_entity_type == "note"]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.APPLE_BOOKS_HIGHLIGHTS_JSON,
                    source_id=self._book_source_id(identity),
                    source_entity_type="book",
                    title=f"{title} by {author}" if author else title,
                    content=f"Apple Books book: {title}" + (f"\nAuthor: {author}" if author else ""),
                    content_type=ContentType.METADATA,
                    metadata={
                        "book_title": title,
                        "author": author,
                        "asset_id": identity[0],
                        "isbn": identity[1],
                        "annotation_count": len(book_annotations),
                        "highlight_count": len(highlights),
                        "note_count": len(notes),
                        "first_annotation_at": created_at.isoformat(),
                        "last_annotation_at": updated_at.isoformat(),
                        "locations": sorted({str(unit.metadata.get("location")) for unit in book_annotations if unit.metadata.get("location")}),
                        "source_files": sorted({str(unit.metadata.get("source_file")) for unit in book_annotations if unit.metadata.get("source_file")}),
                        "annotation_source_ids": [unit.source_id for unit in book_annotations],
                    },
                    tags=["apple_books", "book"],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _book_annotation_edges(self, books: list[KnowledgeUnit], annotations: list[KnowledgeUnit], allowed: set[str]) -> list[KnowledgeEdge]:
        book_ids = {self._book_identity(book.metadata): book.source_id for book in books}
        edges: list[KnowledgeEdge] = []
        for annotation in annotations:
            if annotation.source_entity_type not in allowed:
                continue
            book_id = book_ids.get(self._book_identity(annotation.metadata))
            if book_id:
                edges.append(self._edge(book_id, annotation.source_id, f"book_contains_{annotation.source_entity_type}"))
        return list({edge.id: edge for edge in edges}.values())

    def _book_identity(self, metadata: dict[str, Any]) -> tuple[str, str, str]:
        record = metadata.get("record") if isinstance(metadata.get("record"), dict) else {}
        asset_id = str(metadata.get("asset_id") or "") or first(record, "assetId", "asset_id", "bookId", "book_id")
        isbn = str(metadata.get("isbn") or "") or first(record, "isbn", "ISBN", "isbn13", "ISBN13")
        title = " ".join(str(metadata.get("book_title") or "").casefold().split())
        author = " ".join(str(metadata.get("author") or "").casefold().split())
        fallback = "|".join(part for part in (title, author) if part)
        return asset_id, isbn, fallback

    def _book_source_id(self, identity: tuple[str, str, str]) -> str:
        return digest_source_id("apple_books_highlights_json:book", *identity)

    def _edge(self, from_id: str, to_id: str, relation_type: str) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=digest_source_id("apple-books-highlights-json-edge", from_id, to_id, relation_type),
            from_unit_id=from_id,
            to_unit_id=to_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.APPLE_BOOKS_HIGHLIGHTS_JSON.value,
                "relation_type": relation_type,
            },
        )

    def _title(self, entity_type: str, book_title: str, text: str) -> str:
        label = "Apple Books note" if entity_type == "note" else "Apple Books highlight"
        return f"{label}: {book_title}" if book_title else f"{label}: {text[:80]}"

    def _content(self, book_title: str, author: str, location: str, selected_text: str, note_text: str) -> str:
        parts: list[str] = []
        if book_title:
            parts.append(f"Book: {book_title}")
        if author:
            parts.append(f"Author: {author}")
        if location:
            parts.append(f"Location: {location}")
        if selected_text:
            parts.append(f"\nHighlight:\n{selected_text}")
        if note_text:
            parts.append(f"\nNote:\n{note_text}")
        return "\n".join(parts)
