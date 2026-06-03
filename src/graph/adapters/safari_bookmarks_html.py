"""Adapter for Safari/Netscape bookmarks HTML exports."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SafariBookmarksHtmlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "safari_bookmarks_html"

    @property
    def entity_types(self) -> list[str]:
        return ["bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "bookmark" not in entity_types:
            return result
        sync_at = _ensure_utc(since.last_sync_at) if since else None
        for path in _iter_paths(self.path):
            try:
                bookmarks = _parse(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError):
                continue
            for index, bookmark in enumerate(bookmarks):
                unit = self._unit(bookmark, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, bookmark: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        url = _text(bookmark.get("href"))
        title = _text(bookmark.get("title")) or url
        if not url and not title:
            return None
        add_date = _parse_ts(bookmark.get("add_date"))
        last_modified = _parse_ts(bookmark.get("last_modified"))
        now = datetime.now(timezone.utc)
        metadata = {
            "url": url,
            "folder_path": bookmark.get("folder_path") or [],
            "add_date": add_date.isoformat() if add_date else _text(bookmark.get("add_date")),
            "last_modified": last_modified.isoformat() if last_modified else _text(bookmark.get("last_modified")),
            "icon": _text(bookmark.get("icon")),
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project="safari_bookmarks_html",
            source_id=_source_id(url or title, index),
            source_entity_type="bookmark",
            title=title or "Safari bookmark",
            content="\n".join(part for part in [title, f"URL: {url}" if url else ""] if part),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=["safari", "bookmark"],
            created_at=add_date or last_modified or now,
            updated_at=last_modified or add_date or now,
        )


class _BookmarkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.bookmarks: list[dict[str, Any]] = []
        self.folder_stack: list[str] = []
        self._capture: str | None = None
        self._attrs: dict[str, str] = {}
        self._buffer: list[str] = []
        self._pending_folder = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key.casefold(): value or "" for key, value in attrs}
        if tag.casefold() == "h3":
            self._capture = "folder"
            self._buffer = []
        elif tag.casefold() == "a":
            self._capture = "bookmark"
            self._attrs = attr
            self._buffer = []
        elif tag.casefold() == "dl" and self._pending_folder:
            self._pending_folder = False

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "h3" and self._capture == "folder":
            folder = _text(" ".join(self._buffer))
            if folder:
                self.folder_stack.append(folder)
                self._pending_folder = True
            self._capture = None
        elif tag == "a" and self._capture == "bookmark":
            self.bookmarks.append({**self._attrs, "title": _text(" ".join(self._buffer)), "folder_path": list(self.folder_stack)})
            self._capture = None
            self._attrs = {}
        elif tag == "dl" and self.folder_stack and not self._pending_folder:
            self.folder_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buffer.append(data)


def _parse(text: str) -> list[dict[str, Any]]:
    parser = _BookmarkParser()
    parser.feed(text)
    return parser.bookmarks


def _iter_paths(path: str) -> list[Path]:
    if not path:
        return []
    root = Path(path).expanduser()
    if root.is_file() and root.suffix.lower() in {".html", ".htm"}:
        return [root]
    if root.is_dir():
        return sorted(child for child in root.rglob("*") if child.is_file() and child.suffix.lower() in {".html", ".htm"})
    return []


def _parse_ts(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.fromtimestamp(float(text), tz=timezone.utc)
    except ValueError:
        try:
            return _ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None


def _ensure_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _source_id(identity: str, index: int) -> str:
    digest = hashlib.sha256((identity or str(index)).encode("utf-8")).hexdigest()[:24]
    return f"safari_bookmarks_html:{digest}"


def _text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())
