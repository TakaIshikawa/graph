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
    description: str = ""


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
        self._pending_bookmark: _Bookmark | None = None
        self._capturing_description = False
        self._description_parts: list[str] = []

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

        if tag == "dt":
            # DT starts a new definition term, finalize any description capture
            self._finalize_description()
            return

        if tag == "a":
            # Finalize any pending bookmark before starting a new one
            if self._pending_bookmark is not None:
                self.bookmarks.append(self._pending_bookmark)
                self._pending_bookmark = None
            self._current_link = attrs_dict
            self._link_parts = []
            return

        if tag == "dd":
            # DD following an A tag is a description for that bookmark
            if self._pending_bookmark is not None:
                self._capturing_description = True
                self._description_parts = []

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
                # Create bookmark but keep it pending to potentially receive description
                self._pending_bookmark = _Bookmark(
                    title=title,
                    url=url,
                    folder_path=tuple(self._folders),
                    add_date=self._current_link.get("add_date", "").strip(),
                    last_modified=self._current_link.get("last_modified", "").strip(),
                )
            self._current_link = None
            self._link_parts = []
            return

        if tag == "dd" and self._capturing_description:
            description = self._clean_text(" ".join(self._description_parts))
            if self._pending_bookmark is not None:
                self._pending_bookmark.description = description
            self._capturing_description = False
            self._description_parts = []
            return

        if tag == "dl":
            # Finalize description and any pending bookmark when closing a DL
            self._finalize_description()
            if self._pending_bookmark is not None:
                self.bookmarks.append(self._pending_bookmark)
                self._pending_bookmark = None
            if self._folders:
                self._folders.pop()

    def handle_data(self, data: str) -> None:
        if self._current_link is not None:
            self._link_parts.append(data)
        elif self._capturing_folder:
            self._folder_parts.append(data)
        elif self._capturing_description:
            self._description_parts.append(data)

    def _clean_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def _finalize_description(self) -> None:
        """Finalize any description being captured and attach to pending bookmark."""
        if self._capturing_description:
            description = self._clean_text(" ".join(self._description_parts))
            if self._pending_bookmark is not None and description:
                self._pending_bookmark.description = description
            self._capturing_description = False
            self._description_parts = []


class BookmarksAdapter(SourceAdapter):
    source_project = SourceProject.BOOKMARKS

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
            metadata = {
                "url": bookmark.url,
                "folder_path": folder_path,
                "add_date": bookmark.add_date,
                "last_modified": bookmark.last_modified,
            }
            if bookmark.description:
                metadata["description"] = bookmark.description
            result.units.append(
                KnowledgeUnit(
                    source_project=self.source_project,
                    source_id=bookmark.url,
                    source_entity_type="bookmark",
                    title=bookmark.title,
                    content=self._content(bookmark, folder_path),
                    content_type=ContentType.ARTIFACT,
                    metadata=metadata,
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
        if bookmark.description:
            lines.append(bookmark.description)
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
