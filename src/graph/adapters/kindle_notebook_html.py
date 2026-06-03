"""Adapter for Kindle Notebook HTML exports."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class KindleNotebookHtmlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "kindle_notebook_html"

    @property
    def entity_types(self) -> list[str]:
        return ["highlight", "note"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types or self.entity_types)
        if not requested.intersection(self.entity_types):
            return result
        sync_at = _ensure_utc(since.last_sync_at) if since else None
        for path in _iter_paths(self.path):
            try:
                records = _parse_html(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError):
                continue
            for index, record in enumerate(records):
                units = self._units(record, path.name, index)
                for unit in units:
                    if unit.source_entity_type not in requested:
                        continue
                    if sync_at and unit.updated_at <= sync_at:
                        continue
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.metadata.get("book_title", ""), unit.source_id))
        return result

    def _units(self, record: dict[str, Any], source_file: str, index: int) -> list[KnowledgeUnit]:
        book_title = _clean(record.get("book_title"))
        author = _clean(record.get("author"))
        highlight = _clean(record.get("highlight_text"))
        note = _clean(record.get("note_text"))
        page = _clean(record.get("page"))
        location = _clean(record.get("location"))
        date_text = _clean(record.get("date_text"))
        highlighted_at = _parse_date(date_text)
        now = datetime.now(timezone.utc)
        when = highlighted_at or now
        common = {
            "book_title": book_title,
            "author": author,
            "page": page,
            "location": location,
            "highlight_date": highlighted_at.isoformat() if highlighted_at else date_text,
            "source_file": source_file,
            "entry_index": index,
        }
        units: list[KnowledgeUnit] = []
        for entity_type, text in (("highlight", highlight), ("note", note)):
            if not text:
                continue
            metadata = {**common, "highlight_text": highlight, "note_text": note}
            units.append(
                KnowledgeUnit(
                    source_project="kindle_notebook_html",
                    source_id=_source_id(entity_type, book_title, author, page, location, date_text, text, index),
                    source_entity_type=entity_type,
                    title=_title(entity_type, book_title, text),
                    content=_content(book_title, author, page, location, entity_type, text),
                    content_type=ContentType.INSIGHT if entity_type == "note" else ContentType.ARTIFACT,
                    metadata={key: value for key, value in metadata.items() if value not in ("", None)},
                    tags=["kindle", entity_type],
                    created_at=when,
                    updated_at=when,
                )
            )
        return units


class _KindleNotebookParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.book_title = ""
        self.author = ""
        self.records: list[dict[str, str]] = []
        self._class_stack: list[str] = []
        self._capture: str | None = None
        self._capture_depth = 0
        self._buffer: list[str] = []
        self._current: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = dict(attrs).get("class") or ""
        self._class_stack.append(classes)
        if self._capture is not None:
            self._capture_depth += 1
            return
        if "bookTitle" in classes:
            self._start_capture("book_title")
        elif "authors" in classes:
            self._start_capture("author")
        elif "sectionHeading" in classes:
            self._finish_current()
            self._current = {"heading": ""}
            self._start_capture("heading")
        elif "noteHeading" in classes:
            self._start_capture("note_heading")
        elif "highlight" in classes:
            self._start_capture("highlight_text")
        elif "noteText" in classes:
            self._start_capture("note_text")

    def handle_endtag(self, tag: str) -> None:
        if self._capture is not None:
            if self._capture_depth > 0:
                self._capture_depth -= 1
                if self._class_stack:
                    self._class_stack.pop()
                return
            text = _clean(" ".join(self._buffer))
            if text:
                if self._capture == "book_title":
                    self.book_title = text
                elif self._capture == "author":
                    self.author = re.sub(r"^\s*by\s+", "", text, flags=re.IGNORECASE)
                elif self._current is not None:
                    self._current[self._capture] = text
            self._capture = None
            self._buffer = []
        if self._class_stack:
            self._class_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._finish_current()

    def _start_capture(self, field: str) -> None:
        self._capture = field
        self._capture_depth = 0
        self._buffer = []

    def _finish_current(self) -> None:
        if not self._current:
            return
        heading = self._current.get("heading") or self._current.get("note_heading") or ""
        self._current.update(_parse_heading(heading))
        if self._current.get("highlight_text") or self._current.get("note_text"):
            self._current.setdefault("book_title", self.book_title)
            self._current.setdefault("author", self.author)
            self.records.append(self._current)
        self._current = None


def _parse_html(text: str) -> list[dict[str, str]]:
    parser = _KindleNotebookParser()
    parser.feed(text)
    parser.close()
    return parser.records


def _parse_heading(text: str) -> dict[str, str]:
    page = ""
    location = ""
    date_text = ""
    page_match = re.search(r"\bpage\s+([^\|]+)", text, re.IGNORECASE)
    if page_match:
        page = page_match.group(1).strip()
    location_match = re.search(r"\blocation\s+([^\|]+)", text, re.IGNORECASE)
    if location_match:
        location = location_match.group(1).strip()
    date_match = re.search(r"\|\s*(?:added on\s*)?(.+)$", text, re.IGNORECASE)
    if date_match:
        date_text = date_match.group(1).strip()
    return {"page": page, "location": location, "date_text": date_text}


def _parse_date(value: str) -> datetime | None:
    text = _clean(value)
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return _ensure_utc(datetime.fromisoformat(candidate))
        except ValueError:
            pass
    for fmt in ("%A, %B %d, %Y %I:%M:%S %p", "%B %d, %Y %I:%M:%S %p", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _iter_paths(path: str) -> list[Path]:
    if not path:
        return []
    root = Path(path).expanduser()
    if root.is_file() and root.suffix.lower() in {".html", ".htm"}:
        return [root]
    if root.is_dir():
        return sorted(child for child in root.rglob("*") if child.is_file() and child.suffix.lower() in {".html", ".htm"})
    return []


def _ensure_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _source_id(*parts: Any) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:24]
    return f"kindle_notebook_html:{digest}"


def _title(entity_type: str, book_title: str, text: str) -> str:
    label = "Kindle note" if entity_type == "note" else "Kindle highlight"
    return f"{label}: {book_title or text[:80]}"


def _content(book_title: str, author: str, page: str, location: str, entity_type: str, text: str) -> str:
    parts = [f"Book: {book_title}" if book_title else "", f"Author: {author}" if author else "", f"Page: {page}" if page else "", f"Location: {location}" if location else "", "", text]
    return "\n".join(part for part in parts if part != "")


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
