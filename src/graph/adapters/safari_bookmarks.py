"""Adapter for Safari Bookmarks.plist exports."""

from __future__ import annotations

import hashlib
import plistlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


@dataclass
class _SafariBookmark:
    title: str
    url: str
    folder_path: tuple[str, ...]
    uuid: str
    source_id: str
    created_value: Any
    updated_value: Any


class SafariBookmarksAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "safari_bookmarks"

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

        try:
            with path.open("rb") as handle:
                data = plistlib.load(handle)
        except (OSError, plistlib.InvalidFileException, ValueError) as exc:
            raise ValueError(f"Could not read Safari bookmarks plist: {path}") from exc

        if not isinstance(data, dict):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for bookmark in self._bookmarks(data):
            created_at = self._parse_datetime(bookmark.created_value)
            updated_at = self._parse_datetime(bookmark.updated_value)
            comparable_at = updated_at or created_at
            if sync_at and comparable_at and comparable_at <= sync_at:
                continue
            result.units.append(self._unit_from_bookmark(bookmark, path.name, created_at, updated_at))

        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _bookmarks(self, root: dict[str, Any]) -> list[_SafariBookmark]:
        bookmarks: list[_SafariBookmark] = []
        self._walk(root, (), bookmarks)
        return bookmarks

    def _walk(
        self,
        node: dict[str, Any],
        folders: tuple[str, ...],
        bookmarks: list[_SafariBookmark],
    ) -> None:
        url = self._first(node, "URLString", "URL", "url")
        if not url and isinstance(node.get("URIDictionary"), dict):
            url = self._first(node["URIDictionary"], "URLString", "URL", "url")
        if url:
            title = self._title(node) or url
            bookmarks.append(
                _SafariBookmark(
                    title=title,
                    url=url,
                    folder_path=folders,
                    uuid=self._first(node, "WebBookmarkUUID", "UUID", "uuid"),
                    source_id=self._first(
                        node,
                        "WebBookmarkIdentifier",
                        "WebBookmarkUUID",
                        "UUID",
                        "id",
                    ),
                    created_value=self._first_value(
                        node,
                        "DateAdded",
                        "dateAdded",
                        "Created",
                        "created",
                        "CreationDate",
                    ),
                    updated_value=self._first_value(
                        node,
                        "DateLastModified",
                        "LastModified",
                        "dateLastModified",
                        "Modified",
                        "modified",
                    ),
                )
            )
            return

        next_folders = folders
        folder_title = self._title(node)
        if folder_title:
            next_folders = (*folders, folder_title)

        children = node.get("Children")
        if not isinstance(children, list):
            return
        for child in children:
            if isinstance(child, dict):
                self._walk(child, next_folders, bookmarks)

    def _unit_from_bookmark(
        self,
        bookmark: _SafariBookmark,
        source_file: str,
        created_at: datetime | None,
        updated_at: datetime | None,
    ) -> KnowledgeUnit:
        folder_path = "/".join(bookmark.folder_path)
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.SAFARI_BOOKMARKS,
            source_id=self._source_id(bookmark),
            source_entity_type="bookmark",
            title=bookmark.title,
            content=self._content(bookmark, folder_path),
            content_type=ContentType.ARTIFACT,
            metadata={
                "url": bookmark.url,
                "folder_path": folder_path,
                "title": bookmark.title,
                "uuid": bookmark.uuid,
                "safari_id": bookmark.source_id,
                "created_at": self._metadata_datetime(bookmark.created_value),
                "updated_at": self._metadata_datetime(bookmark.updated_value),
                "source_file": source_file,
            },
            tags=self._folder_tags(bookmark.folder_path),
            created_at=created_at or updated_at or now,
            updated_at=updated_at or created_at or now,
        )

    def _title(self, node: dict[str, Any]) -> str:
        title = self._first(node, "Title", "title")
        uri_dictionary = node.get("URIDictionary")
        if title or not isinstance(uri_dictionary, dict):
            return title
        return self._first(uri_dictionary, "title", "Title")

    def _source_id(self, bookmark: _SafariBookmark) -> str:
        if bookmark.source_id:
            return f"safari_bookmarks:{bookmark.source_id}"
        digest = hashlib.sha256(bookmark.url.encode("utf-8")).hexdigest()
        return f"url:{digest[:24]}"

    def _content(self, bookmark: _SafariBookmark, folder_path: str) -> str:
        lines = [bookmark.title, f"URL: {bookmark.url}"]
        if folder_path:
            lines.append(f"Folder: {folder_path}")
        return "\n".join(lines)

    def _folder_tags(self, folder_path: tuple[str, ...]) -> list[str]:
        tags: list[str] = []
        for index in range(len(folder_path)):
            tag = "/".join(folder_path[: index + 1])
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        value = self._first_value(item, *keys)
        if value is None or isinstance(value, (dict, list, bytes, bytearray)):
            return ""
        text = re.sub(r"\s+", " ", str(value)).strip()
        return text

    def _first_value(self, item: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = item.get(key)
            if value is not None:
                return value
        return None

    def _metadata_datetime(self, value: Any) -> str:
        parsed = self._parse_datetime(value)
        if parsed is not None:
            return parsed.isoformat()
        if value is None or isinstance(value, (dict, list, bytes, bytearray)):
            return ""
        return str(value).strip()

    def _parse_datetime(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, (int, float)):
            try:
                parsed = datetime.fromtimestamp(value, tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        elif isinstance(value, str) and value.strip():
            text = value.strip()
            if re.fullmatch(r"\d+(?:\.0+)?", text):
                try:
                    parsed = datetime.fromtimestamp(int(float(text)), tz=timezone.utc)
                except (OSError, OverflowError, ValueError):
                    return None
            else:
                try:
                    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                except ValueError:
                    return None
        else:
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        parsed = since.last_sync_at
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
