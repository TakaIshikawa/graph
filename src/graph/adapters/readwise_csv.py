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
        allowed = set(entity_types) if entity_types else {"highlight"}
        if not allowed.intersection(self.entity_types):
            return result

        sync_at = self._sync_datetime(since) if since else None
        documents: dict[str, dict[str, Any]] = {}
        document_units: dict[str, KnowledgeUnit] = {}
        edges: list[KnowledgeEdge] = []
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
                document_source_id = self._document_source_id_from_row(row)
                unit.metadata["document_source_id"] = document_source_id
                self._accumulate_document(documents, document_source_id, row, source_file)
                if "highlight" in allowed:
                    result.units.append(unit)
                if "document" in allowed and "highlight" in allowed:
                    edges.append(self._document_edge(document_source_id, unit.source_id, unit.updated_at))

        if "document" in allowed:
            for source_id, info in documents.items():
                document_units[source_id] = self._document_unit(source_id, info)
            result.units.extend(document_units.values())
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.extend(sorted(edges, key=lambda edge: edge.id))
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

    def _document_source_id_from_row(self, row: dict[str, Any]) -> str:
        title = self._first(row, "Book Title", "Title", "Article Title", "Document Title")
        author = self._first(row, "Book Author", "Author", "Authors")
        url = self._first(row, "URL", "Source URL", "Book URL", "Article URL")
        category = self._first(row, "Category")
        if url:
            stable = f"url:{url.strip().casefold()}"
        else:
            stable = "\n".join([title.casefold(), author.casefold(), category.casefold()])
        digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
        return f"readwise_csv:document:{digest}"

    def _accumulate_document(
        self,
        documents: dict[str, dict[str, Any]],
        source_id: str,
        row: dict[str, Any],
        source_file: str,
    ) -> None:
        info = documents.setdefault(
            source_id,
            {
                "title": "",
                "author": "",
                "url": "",
                "category": "",
                "source_files": set(),
                "highlight_count": 0,
                "created_at": None,
                "updated_at": None,
            },
        )
        for key, keys in {
            "title": ("Book Title", "Title", "Article Title", "Document Title"),
            "author": ("Book Author", "Author", "Authors"),
            "url": ("URL", "Source URL", "Book URL", "Article URL"),
            "category": ("Category",),
        }.items():
            if not info[key]:
                info[key] = self._first(row, *keys)
        info["source_files"].add(source_file)
        info["highlight_count"] += 1
        highlighted_at = self._parse_datetime(self._first(row, "Highlighted at", "Highlighted At", "Date"))
        if highlighted_at is not None:
            current_created = info["created_at"]
            current_updated = info["updated_at"]
            info["created_at"] = highlighted_at if current_created is None else min(current_created, highlighted_at)
            info["updated_at"] = highlighted_at if current_updated is None else max(current_updated, highlighted_at)

    def _document_unit(self, source_id: str, info: dict[str, Any]) -> KnowledgeUnit:
        now = datetime.now(timezone.utc)
        title = info["title"] or info["url"] or "Untitled Readwise document"
        metadata = {
            "title": info["title"],
            "author": info["author"],
            "url": info["url"],
            "category": info["category"],
            "source_files": sorted(info["source_files"]),
            "highlight_count": info["highlight_count"],
        }
        return KnowledgeUnit(
            source_project=SourceProject.READWISE_CSV,
            source_id=source_id,
            source_entity_type="document",
            title=title,
            content=self._document_content(title, info),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["readwise", "document", *([info["category"]] if info["category"] else [])],
            created_at=info["created_at"] or now,
            updated_at=info["updated_at"] or now,
        )

    def _document_content(self, title: str, info: dict[str, Any]) -> str:
        parts = [title]
        if info["author"]:
            parts.append(f"Author: {info['author']}")
        if info["url"]:
            parts.append(f"URL: {info['url']}")
        if info["category"]:
            parts.append(f"Category: {info['category']}")
        parts.append(f"Highlights: {info['highlight_count']}")
        return "\n".join(parts)

    def _document_edge(self, document_source_id: str, highlight_source_id: str, created_at: datetime) -> KnowledgeEdge:
        digest = hashlib.sha256(f"{document_source_id}|{highlight_source_id}|contains".encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(
            id=f"readwise_csv:contains:{digest}",
            from_unit_id=document_source_id,
            to_unit_id=highlight_source_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.READWISE_CSV.value,
                "relation_type": "document_contains_highlight",
            },
            created_at=created_at,
        )

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
