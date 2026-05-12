"""Adapter for Readwise highlight CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class ReadwiseCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "readwise_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["highlight", "document"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or ["highlight"])
        if not allowed.intersection(self.entity_types):
            return result

        sync_at = self._sync_datetime(since) if since else None
        document_rows: dict[str, dict[str, Any]] = {}
        document_counts: dict[str, int] = {}
        document_files: dict[str, set[str]] = {}
        edge_candidates: list[KnowledgeEdge] = []
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
                document_id = self._document_source_id_from_metadata(unit.metadata)
                unit.metadata["document_source_id"] = document_id
                document_rows.setdefault(document_id, unit.metadata)
                document_counts[document_id] = document_counts.get(document_id, 0) + 1
                document_files.setdefault(document_id, set()).add(source_file)
                if "highlight" in allowed:
                    result.units.append(unit)
                if {"highlight", "document"}.issubset(allowed):
                    edge_candidates.append(self._document_edge(document_id, unit.source_id))

        if "document" in allowed:
            for document_id in sorted(document_rows):
                result.units.append(self._document_unit(document_id, document_rows[document_id], document_counts[document_id], sorted(document_files[document_id])))
        result.edges.extend(edge_candidates)

        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[tuple[Path, str]]:
        entries: list[tuple[Path, str]] = []
        for source in re.split(r"[\n,]", self.path):
            source = source.strip()
            if not source:
                continue
            path = Path(source).expanduser()
            if path.is_dir():
                for child in sorted(path.rglob("*.csv")):
                    if child.is_file():
                        entries.append((child, self._relative_path(child, path)))
            elif path.exists() and path.is_file():
                entries.append((path, path.name))
        return entries

    def _read_rows(self, path: Path) -> list[tuple[int, dict[str, Any]]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []

            rows: list[tuple[int, dict[str, Any]]] = []
            for row_number, row in enumerate(reader, start=2):
                normalized = {
                    str(key).strip(): value
                    for key, value in row.items()
                    if key is not None
                }
                if any(self._text(value) for value in normalized.values()):
                    rows.append((row_number, normalized))
            return rows

    def _unit_from_row(
        self,
        row: dict[str, Any],
        source_file: str,
        row_number: int,
    ) -> KnowledgeUnit | None:
        text = self._first(row, "Highlight", "Text", "Highlighted Text", "Quote")
        note = self._first(row, "Note", "Notes", "Annotation")
        if not text and not note:
            return None

        title = self._first(row, "Book Title", "Title", "Article Title", "Document Title")
        author = self._first(row, "Book Author", "Author", "Authors")
        url = self._first(row, "URL", "Source URL", "Book URL", "Article URL")
        category = self._first(row, "Category")
        location = self._first(row, "Location", "Location Type", "Position")
        highlighted_text = self._first(row, "Highlighted at", "Highlighted At", "Date")
        highlighted_at = self._parse_datetime(highlighted_text)
        tags = self._parse_tags(self._first(row, "Tags", "Tag"))
        now = datetime.now(timezone.utc)

        metadata = {
            "source_file": source_file,
            "row_number": row_number,
            "title": title,
            "author": author,
            "url": url,
            "category": category,
            "location": location,
            "note": note,
            "text": text,
            "tags": tags,
            "highlighted_at": highlighted_text,
        }

        return KnowledgeUnit(
            source_project=SourceProject.READWISE_CSV,
            source_id=self._source_id(row, title, author, text, note, url, location, highlighted_text),
            source_entity_type="highlight",
            title=title or "Readwise highlight",
            content=self._content(text, note, title, author, url, location, category, tags),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=tags,
            created_at=highlighted_at or now,
            updated_at=highlighted_at or now,
        )

    def _document_unit(
        self,
        source_id: str,
        metadata: dict[str, Any],
        highlight_count: int,
        source_files: list[str],
    ) -> KnowledgeUnit:
        title = str(metadata.get("title") or "Readwise document")
        author = str(metadata.get("author") or "")
        url = str(metadata.get("url") or "")
        category = str(metadata.get("category") or "")
        document_metadata = {
            "title": title if title != "Readwise document" else "",
            "author": author,
            "url": url,
            "category": category,
            "source_files": source_files,
            "highlight_count": highlight_count,
        }
        return KnowledgeUnit(
            source_project=SourceProject.READWISE_CSV,
            source_id=source_id,
            source_entity_type="document",
            title=title,
            content=self._document_content(title, author, url, category, highlight_count),
            content_type=ContentType.METADATA,
            metadata=document_metadata,
            tags=[category] if category else [],
        )

    def _document_source_id_from_metadata(self, metadata: dict[str, Any]) -> str:
        return self._document_source_id(
            str(metadata.get("title") or ""),
            str(metadata.get("author") or ""),
            str(metadata.get("url") or ""),
            str(metadata.get("category") or ""),
        )

    def _document_source_id(self, title: str, author: str, url: str, category: str) -> str:
        digest = hashlib.sha256("\n".join([title, author, url, category]).encode("utf-8")).hexdigest()[:24]
        return f"readwise_csv:document:{digest}"

    def _document_edge(self, document_source_id: str, highlight_source_id: str) -> KnowledgeEdge:
        digest = hashlib.sha256(f"{document_source_id}|{highlight_source_id}|contains".encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(
            id=f"readwise-csv-document-contains-{digest}",
            from_unit_id=document_source_id,
            to_unit_id=highlight_source_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.READWISE_CSV.value,
                "relation_type": "document_contains_highlight",
            },
        )

    def _document_content(self, title: str, author: str, url: str, category: str, highlight_count: int) -> str:
        parts = [title]
        if author:
            parts.append(f"Author: {author}")
        if url:
            parts.append(f"URL: {url}")
        if category:
            parts.append(f"Category: {category}")
        parts.append(f"Highlights: {highlight_count}")
        return "\n".join(parts)

    def _source_id(
        self,
        row: dict[str, Any],
        title: str,
        author: str,
        text: str,
        note: str,
        url: str,
        location: str,
        highlighted_at: str,
    ) -> str:
        highlight_id = self._first(row, "Highlight ID", "ID", "Id", "Readwise ID")
        if highlight_id:
            return f"readwise_csv:{highlight_id}"

        digest = hashlib.sha256(
            "\n".join([title, author, text, note, url, location, highlighted_at]).encode(
                "utf-8"
            )
        ).hexdigest()
        return f"readwise_csv:{digest[:24]}"

    def _content(
        self,
        text: str,
        note: str,
        title: str,
        author: str,
        url: str,
        location: str,
        category: str,
        tags: list[str],
    ) -> str:
        parts: list[str] = []
        if text:
            parts.append(text)
        if note:
            parts.append(f"Note: {note}")
        if title:
            parts.append(f"Title: {title}")
        if author:
            parts.append(f"Author: {author}")
        if url:
            parts.append(f"URL: {url}")
        if location:
            parts.append(f"Location: {location}")
        if category:
            parts.append(f"Category: {category}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _parse_tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for tag in re.split(r"[,;|]", value):
            normalized = re.sub(r"\s+", " ", tag.strip().removeprefix("#")).strip()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = self._value(row, key)
            if value:
                return value
        return ""

    def _value(self, row: dict[str, Any], wanted: str) -> str:
        wanted_key = self._normalize_key(wanted)
        for key, value in row.items():
            if self._normalize_key(key) == wanted_key:
                return self._text(value)
        return ""

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")

    def _text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        cleaned = value.strip()
        if re.fullmatch(r"\d+(?:\.0+)?", cleaned):
            try:
                return datetime.fromtimestamp(int(float(cleaned)), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        if cleaned.endswith(" UTC"):
            cleaned = cleaned.removesuffix(" UTC") + "+00:00"
        try:
            parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _relative_path(self, path: Path, root: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()
