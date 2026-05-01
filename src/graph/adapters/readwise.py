"""Adapter for local Readwise highlight JSON exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class ReadwiseAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "readwise"

    @property
    def entity_types(self) -> list[str]:
        return ["highlight"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "highlight" not in entity_types:
            return result

        paths = self._json_paths()
        if not paths:
            return result

        sync_at = self._sync_datetime(since) if since else None
        for path in paths:
            for highlight in self._read_highlights(path):
                unit = self._unit_from_highlight(highlight)
                if unit is None:
                    continue
                comparable_at = self._comparable_datetime(unit)
                if sync_at and comparable_at and comparable_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _json_paths(self) -> list[Path]:
        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists():
            return []
        if path.is_file():
            return [path] if path.suffix.lower() == ".json" else []
        if path.is_dir():
            return sorted(item for item in path.rglob("*.json") if item.is_file())
        return []

    def _read_highlights(self, path: Path) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        return self._highlight_records(parsed)

    def _highlight_records(
        self, value: Any, parent: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if isinstance(value, list):
            records: list[dict[str, Any]] = []
            for item in value:
                records.extend(self._highlight_records(item, parent))
            return records
        if not isinstance(value, dict):
            return []

        parent_context = dict(parent or {})
        for key, item in value.items():
            if key not in {"highlights", "results"}:
                parent_context.setdefault(key, item)

        records = []
        for key in ("results", "highlights"):
            nested = value.get(key)
            if isinstance(nested, list):
                for item in nested:
                    records.extend(self._highlight_records(item, parent_context))
            elif isinstance(nested, dict):
                for item in nested.values():
                    records.extend(self._highlight_records(item, parent_context))
        if records:
            return records

        merged = dict(parent or {})
        merged.update(value)
        if self._has_highlight_content(merged):
            return [merged]
        return []

    def _unit_from_highlight(self, highlight: dict[str, Any]) -> KnowledgeUnit | None:
        text = self._highlight_text(highlight)
        note = self._first(highlight, "note", "notes", "annotation")
        if not text and not note:
            return None

        title = self._title(highlight)
        author = self._author(highlight)
        book_id = self._book_id(highlight)
        source_url = self._source_url(highlight)
        location = self._first(highlight, "location", "location_start", "position")
        created_text = self._first(highlight, "created_at", "created", "date_added")
        updated_text = self._first(highlight, "updated_at", "updated", "last_updated_at")
        highlighted_text = self._first(
            highlight, "highlighted_at", "highlighted_date", "highlighted_on"
        )
        created_at = self._parse_datetime(created_text)
        updated_at = self._parse_datetime(updated_text)
        highlighted_at = self._parse_datetime(highlighted_text)
        tags = self._tags(highlight.get("tags") or highlight.get("tag"))

        metadata = {
            "id": self._first(highlight, "id", "highlight_id"),
            "book_id": book_id,
            "readwise_url": self._first(highlight, "readwise_url"),
            "source_url": source_url,
            "url": self._first(highlight, "url"),
            "title": title,
            "author": author,
            "category": self._first(highlight, "category"),
            "source": self._first(highlight, "source"),
            "location": location,
            "location_type": self._first(highlight, "location_type"),
            "note": note,
            "text": text,
            "tags": tags,
            "created_at": created_text,
            "updated_at": updated_text,
            "highlighted_at": highlighted_text,
            "book": self._jsonable(highlight.get("book")),
        }

        return KnowledgeUnit(
            source_project=SourceProject.READWISE,
            source_id=self._source_id(highlight, book_id, title, text, note, location),
            source_entity_type="highlight",
            title=title or "Readwise highlight",
            content=self._content(text, note, title, author, source_url, location, tags),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=tags,
            created_at=created_at or highlighted_at or updated_at or datetime.now(timezone.utc),
            updated_at=updated_at or highlighted_at or created_at or datetime.now(timezone.utc),
        )

    def _has_highlight_content(self, value: dict[str, Any]) -> bool:
        note = self._first(value, "note", "notes", "annotation")
        return bool(self._highlight_text(value) or note)

    def _highlight_text(self, highlight: dict[str, Any]) -> str:
        return self._first(
            highlight,
            "text",
            "highlighted_text",
            "highlight",
            "quote",
            "content",
            "highlight_text",
        )

    def _title(self, highlight: dict[str, Any]) -> str:
        book = highlight.get("book")
        if isinstance(book, dict):
            title = self._first(book, "title", "name")
            if title:
                return title
        return self._first(highlight, "title", "book_title", "article_title", "document_title")

    def _author(self, highlight: dict[str, Any]) -> str:
        book = highlight.get("book")
        if isinstance(book, dict):
            author = self._first(book, "author", "authors")
            if author:
                return author
        return self._first(highlight, "author", "authors", "book_author")

    def _book_id(self, highlight: dict[str, Any]) -> str:
        book = highlight.get("book")
        if isinstance(book, dict):
            book_id = self._first(book, "id", "book_id", "asin", "isbn")
            if book_id:
                return book_id
        return self._first(
            highlight,
            "book_id",
            "readwise_book_id",
            "asin",
            "isbn",
            "article_id",
            "document_id",
        )

    def _source_url(self, highlight: dict[str, Any]) -> str:
        book = highlight.get("book")
        if isinstance(book, dict):
            source_url = self._first(book, "source_url", "url")
            if source_url:
                return source_url
        return self._first(highlight, "source_url", "url", "book_url", "article_url")

    def _source_id(
        self,
        highlight: dict[str, Any],
        book_id: str,
        title: str,
        text: str,
        note: str,
        location: str,
    ) -> str:
        highlight_id = self._first(highlight, "id", "highlight_id")
        if highlight_id:
            return f"readwise:{highlight_id}"

        highlighted_at = self._first(highlight, "highlighted_at", "highlighted_date")
        digest = hashlib.sha256(
            "\n".join([book_id, title, text, note, location, highlighted_at]).encode("utf-8")
        ).hexdigest()
        return f"readwise:{digest[:24]}"

    def _content(
        self,
        text: str,
        note: str,
        title: str,
        author: str,
        source_url: str,
        location: str,
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
        if source_url:
            parts.append(f"URL: {source_url}")
        if location:
            parts.append(f"Location: {location}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _tags(self, value: Any) -> list[str]:
        raw_tags: list[Any]
        if isinstance(value, dict):
            if any(key in value for key in ("name", "tag", "text")):
                raw_tags = [value]
            else:
                raw_tags = []
                for key, tag_value in value.items():
                    if isinstance(tag_value, dict):
                        raw_tags.append(tag_value.get("name") or tag_value.get("tag") or key)
                    else:
                        raw_tags.append(tag_value or key)
        elif isinstance(value, list):
            raw_tags = value
        elif isinstance(value, str):
            raw_tags = re.split(r"[,;|]", value)
        else:
            raw_tags = []

        tags: list[str] = []
        for tag in raw_tags:
            if isinstance(tag, dict):
                tag = tag.get("name") or tag.get("tag") or tag.get("text") or ""
            normalized = re.sub(r"\s+", " ", str(tag).strip().removeprefix("#")).strip()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                text = ", ".join(
                    str(part).strip()
                    for part in value
                    if not isinstance(part, (dict, list)) and str(part).strip()
                )
                if text:
                    return text
                continue
            if isinstance(value, dict):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        return str(value)

    def _parse_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        if re.fullmatch(r"\d+(?:\.0+)?", value):
            try:
                return datetime.fromtimestamp(int(float(value)), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
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

    def _comparable_datetime(self, unit: KnowledgeUnit) -> datetime | None:
        highlighted_at = self._parse_datetime(str(unit.metadata.get("highlighted_at") or ""))
        return unit.updated_at or highlighted_at or unit.created_at
