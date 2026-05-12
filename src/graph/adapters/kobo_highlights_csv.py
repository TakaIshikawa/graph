"""Adapter for Kobo highlights CSV exports."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class KoboHighlightsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "kobo_highlights_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["highlight", "note", "book"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types or self.entity_types)
        if not requested.intersection(self.entity_types):
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        annotation_units: list[KnowledgeUnit] = []
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
                annotation_units.append(unit)

        book_units = self._book_units(annotation_units)
        result.units.extend(unit for unit in annotation_units if unit.source_entity_type in requested)
        if "book" in requested:
            result.units.extend(book_units)
        if "book" in requested and requested.intersection({"highlight", "note"}):
            result.edges.extend(self._book_edges(book_units, annotation_units, requested))
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        result.edges.sort(key=lambda edge: edge.id)
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
        highlight = self._first(row, "Annotation", "Highlight", "Highlighted Text", "Text", "annotation", "highlight_text")
        note = self._first(row, "Note", "Notes", "Annotation Note", "note", "note_text")
        if not highlight and not note:
            return None

        entity_type = "note" if note and not highlight else "highlight"
        book_title = self._first(row, "Book Title", "Title", "book_title", "title")
        author = self._first(row, "Author", "Authors", "author")
        isbn = self._clean_isbn(self._first(row, "ISBN", "ISBN13", "isbn", "isbn13"))
        chapter = self._first(row, "Chapter", "Chapter Title", "chapter")
        location = self._first(row, "Location", "Page", "Position", "Chapter Progress", "location", "page")
        color = self._first(row, "Color", "colour", "color")
        created = self._parse_datetime(self._first(row, "Date Created", "Created", "date_created", "created_at"))
        modified = self._parse_datetime(self._first(row, "Date Modified", "Modified", "date_modified", "updated_at"))
        book_url = self._first(row, "Book URL", "URL", "Link", "book_url", "url")
        updated_at = modified or created or datetime.now(timezone.utc)

        metadata = {
            "book_title": book_title,
            "author": author,
            "isbn": isbn,
            "highlight": highlight,
            "note": note,
            "color": color,
            "chapter": chapter,
            "location": location,
            "date_created": created.isoformat() if created else "",
            "date_modified": modified.isoformat() if modified else "",
            "book_url": book_url,
            "source_file": str(path),
            "row": dict(row),
        }
        return KnowledgeUnit(
            source_project=SourceProject.KOBO_HIGHLIGHTS_CSV,
            source_id=self._source_id(book_title, isbn, location, highlight, note, created),
            source_entity_type=entity_type,
            title=self._title(book_title, highlight or note, entity_type),
            content=self._content(book_title, author, highlight, note, chapter, location),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=self._dedupe(["kobo", entity_type, color.lower() if color else ""]),
            created_at=created or updated_at,
            updated_at=updated_at,
        )

    def _source_id(
        self,
        book_title: str,
        isbn: str,
        location: str,
        highlight: str,
        note: str,
        created: datetime | None,
    ) -> str:
        raw = "|".join([isbn or book_title, location, highlight, note, created.isoformat() if created else ""])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"kobo_highlights_csv:{digest}"

    def _title(self, book_title: str, text: str, entity_type: str) -> str:
        prefix = "Kobo note" if entity_type == "note" else "Kobo highlight"
        if book_title:
            return f"{prefix}: {book_title}"
        return f"{prefix}: {text[:80]}"

    def _content(self, book_title: str, author: str, highlight: str, note: str, chapter: str, location: str) -> str:
        parts: list[str] = []
        if book_title:
            parts.append(f"Book: {book_title}")
        if author:
            parts.append(f"Author: {author}")
        if chapter:
            parts.append(f"Chapter: {chapter}")
        if location:
            parts.append(f"Location: {location}")
        if highlight:
            parts.append(f"\nHighlight:\n{highlight}")
        if note:
            parts.append(f"\nNote:\n{note}")
        return "\n".join(parts)

    def _book_units(self, annotations: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        display: dict[str, tuple[str, str, str]] = {}
        for annotation in annotations:
            key = self._book_key(annotation)
            if not key:
                continue
            grouped.setdefault(key, []).append(annotation)
            display.setdefault(
                key,
                (
                    str(annotation.metadata.get("book_title") or ""),
                    str(annotation.metadata.get("author") or ""),
                    str(annotation.metadata.get("isbn") or ""),
                ),
            )

        books: list[KnowledgeUnit] = []
        for key, book_annotations in sorted(grouped.items()):
            unique_annotations = sorted(
                {annotation.source_id: annotation for annotation in book_annotations}.values(),
                key=lambda annotation: annotation.source_id,
            )
            book_title, author, isbn = display[key]
            chapters = sorted(
                {str(annotation.metadata.get("chapter")) for annotation in unique_annotations if annotation.metadata.get("chapter")}
            )
            annotation_source_ids = [annotation.source_id for annotation in unique_annotations]
            highlight_count = sum(1 for annotation in unique_annotations if annotation.source_entity_type == "highlight")
            note_count = sum(1 for annotation in unique_annotations if annotation.source_entity_type == "note")
            source_files = sorted(
                {str(annotation.metadata.get("source_file")) for annotation in unique_annotations if annotation.metadata.get("source_file")}
            )
            title = book_title or author or "Kobo book"
            content = [f"Book: {title}", f"Annotations: {len(unique_annotations)}"]
            if author:
                content.append(f"Author: {author}")
            books.append(
                KnowledgeUnit(
                    source_project=SourceProject.KOBO_HIGHLIGHTS_CSV,
                    source_id=f"kobo_highlights_csv:book:{key}",
                    source_entity_type="book",
                    title=title,
                    content="\n".join(content),
                    content_type=ContentType.METADATA,
                    metadata={
                        "book_title": book_title,
                        "author": author,
                        "isbn": isbn,
                        "annotation_count": len(unique_annotations),
                        "highlight_count": highlight_count,
                        "note_count": note_count,
                        "annotation_source_ids": annotation_source_ids,
                        "chapters": chapters,
                        "source_files": source_files,
                    },
                    tags=self._dedupe(["kobo", "book", author]),
                    created_at=min(annotation.created_at for annotation in unique_annotations),
                    updated_at=max(annotation.updated_at for annotation in unique_annotations),
                )
            )
        return books

    def _book_edges(
        self,
        books: list[KnowledgeUnit],
        annotations: list[KnowledgeUnit],
        requested: set[str],
    ) -> list[KnowledgeEdge]:
        book_ids = {self._book_key_from_metadata(book.metadata): book.source_id for book in books}
        edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str]] = set()
        for annotation in annotations:
            if annotation.source_entity_type not in requested:
                continue
            book_id = book_ids.get(self._book_key(annotation))
            if not book_id or (book_id, annotation.source_id) in seen:
                continue
            seen.add((book_id, annotation.source_id))
            edges.append(
                KnowledgeEdge(
                    id=self._edge_id(book_id, annotation.source_id),
                    from_unit_id=book_id,
                    to_unit_id=annotation.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.KOBO_HIGHLIGHTS_CSV.value,
                        "from_entity_type": "book",
                        "to_entity_type": annotation.source_entity_type,
                        "book_title": annotation.metadata.get("book_title"),
                    },
                    created_at=annotation.created_at,
                )
            )
        return edges

    def _book_key(self, annotation: KnowledgeUnit) -> str:
        return self._book_key_from_metadata(annotation.metadata)

    def _book_key_from_metadata(self, metadata: dict[str, Any]) -> str:
        isbn = str(metadata.get("isbn") or "").strip()
        if isbn:
            raw = f"isbn:{isbn}"
        else:
            title = " ".join(str(metadata.get("book_title") or "").casefold().split())
            author = " ".join(str(metadata.get("author") or "").casefold().split())
            if not title and not author:
                return ""
            raw = f"title-author:{title}|{author}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _edge_id(self, book_id: str, annotation_id: str) -> str:
        digest = hashlib.sha256(f"{book_id}|{annotation_id}|contains".encode("utf-8")).hexdigest()[:24]
        return f"kobo-highlights-book-contains-{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).lower(): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.lower())
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _dedupe(self, values: Any) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _clean_isbn(self, value: str) -> str:
        return value.strip().strip('="').replace("-", "").replace(" ", "")

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        text = value.strip()
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
