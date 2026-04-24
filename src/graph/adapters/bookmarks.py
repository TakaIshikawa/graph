"""Adapter for Netscape bookmark HTML exports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


@dataclass
class _Bookmark:
    title: str
    url: str
    folder_path: tuple[str, ...]
    add_date: str
    last_modified: str


class _BookmarksParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.bookmarks: list[_Bookmark] = []
        self._folders: list[str] = []
        self._pending_folder: str | None = None
        self._capturing_folder = False
        self._folder_parts: list[str] = []
        self._current_link: dict[str, str] | None = None
        self._link_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {key.lower(): value or "" for key, value in attrs}

        if tag == "h3":
            self._capturing_folder = True
            self._folder_parts = []
            return

        if tag == "dl":
            if self._pending_folder is not None:
                self._folders.append(self._pending_folder)
                self._pending_folder = None
            return

        if tag == "a":
            self._current_link = attrs_dict
            self._link_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "h3" and self._capturing_folder:
            folder = self._clean_text(" ".join(self._folder_parts))
            if folder:
                self._pending_folder = folder
            self._capturing_folder = False
            self._folder_parts = []
            return

        if tag == "a" and self._current_link is not None:
            url = self._current_link.get("href", "").strip()
            if url:
                title = self._clean_text(" ".join(self._link_parts)) or url
                self.bookmarks.append(
                    _Bookmark(
                        title=title,
                        url=url,
                        folder_path=tuple(self._folders),
                        add_date=self._current_link.get("add_date", "").strip(),
                        last_modified=self._current_link.get("last_modified", "").strip(),
                    )
                )
            self._current_link = None
            self._link_parts = []
            return

        if tag == "dl" and self._folders:
            self._folders.pop()

    def handle_data(self, data: str) -> None:
        if self._current_link is not None:
            self._link_parts.append(data)
        elif self._capturing_folder:
            self._folder_parts.append(data)

    def _clean_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()


class BookmarksAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "bookmarks"

    @property
    def entity_types(self) -> list[str]:
        return ["bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "bookmark" not in entity_types:
            return result

        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists() or not path.is_file():
            return result

        parser = _BookmarksParser()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))

        sync_at = self._sync_datetime(since) if since else None
        for bookmark in parser.bookmarks:
            created_at = self._parse_unix_datetime(bookmark.add_date)
            modified_at = self._parse_unix_datetime(bookmark.last_modified)
            comparable_at = modified_at or created_at
            if sync_at and comparable_at and comparable_at <= sync_at:
                continue

            folder_path = "/".join(bookmark.folder_path)
            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.BOOKMARKS,
                    source_id=bookmark.url,
                    source_entity_type="bookmark",
                    title=bookmark.title,
                    content=self._content(bookmark, folder_path),
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "url": bookmark.url,
                        "folder_path": folder_path,
                        "add_date": bookmark.add_date,
                        "last_modified": bookmark.last_modified,
                    },
                    tags=self._folder_tags(bookmark.folder_path),
                    created_at=created_at or datetime.now(timezone.utc),
                    updated_at=modified_at or created_at or datetime.now(timezone.utc),
                )
            )

        return result

    def _content(self, bookmark: _Bookmark, folder_path: str) -> str:
        lines = [bookmark.title, bookmark.url]
        if folder_path:
            lines.append(folder_path)
        return "\n".join(lines)

    def _folder_tags(self, folder_path: tuple[str, ...]) -> list[str]:
        tags: list[str] = []
        for index in range(len(folder_path)):
            tag = "/".join(folder_path[: index + 1])
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _parse_unix_datetime(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromtimestamp(int(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
