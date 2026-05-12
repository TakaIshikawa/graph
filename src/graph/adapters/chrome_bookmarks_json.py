"""Adapter for Chrome Bookmarks JSON exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


@dataclass
class _Bookmark:
    title: str
    url: str
    folder_path: tuple[str, ...]
    root: str
    root_name: str
    date_added: Any
    date_last_used: Any
    guid: str
    source_id: str


class ChromeBookmarksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "chrome_bookmarks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["bookmark"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types and "bookmark" not in entity_types:
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                parsed = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for bookmark in self._bookmarks(parsed):
                created_at = self._chrome_datetime(bookmark.date_added)
                last_used_at = self._chrome_datetime(bookmark.date_last_used)
                comparable_at = last_used_at or created_at
                if sync_at and comparable_at and comparable_at <= sync_at:
                    continue
                unit = self._unit(bookmark, path.name, created_at, last_used_at)
                result.units.append(unit)
                edge = self._folder_edge(bookmark, unit.source_id)
                if edge:
                    result.edges.append(edge)

        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.json") if child.is_file())

    def _bookmarks(self, parsed: Any) -> list[_Bookmark]:
        if not isinstance(parsed, dict):
            return []
        roots = parsed.get("roots")
        if not isinstance(roots, dict):
            roots = parsed
        bookmarks: list[_Bookmark] = []
        for root_key, node in roots.items():
            if not isinstance(node, dict):
                continue
            root_name = self._root_name(str(root_key), node)
            self._walk(node, (), str(root_key), root_name, bookmarks)
        return bookmarks

    def _walk(
        self,
        node: dict[str, Any],
        folder_path: tuple[str, ...],
        root: str,
        root_name: str,
        bookmarks: list[_Bookmark],
    ) -> None:
        if node.get("type") == "url" or node.get("url"):
            url = self._text(node.get("url"))
            if not url:
                return
            bookmarks.append(
                _Bookmark(
                    title=self._text(node.get("name")) or url,
                    url=url,
                    folder_path=folder_path,
                    root=root,
                    root_name=root_name,
                    date_added=node.get("date_added"),
                    date_last_used=node.get("date_last_used"),
                    guid=self._text(node.get("guid")),
                    source_id=self._text(node.get("id")),
                )
            )
            return

        name = self._text(node.get("name"))
        next_path = folder_path
        if name and name != root_name:
            next_path = (*folder_path, name)
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    self._walk(child, next_path, root, root_name, bookmarks)

    def _unit(
        self,
        bookmark: _Bookmark,
        source_file: str,
        created_at: datetime | None,
        last_used_at: datetime | None,
    ) -> KnowledgeUnit:
        now = datetime.now(timezone.utc)
        folder_path = "/".join(bookmark.folder_path)
        metadata = {
            "url": bookmark.url,
            "domain": urlparse(bookmark.url).netloc,
            "folder_path": folder_path,
            "root": bookmark.root,
            "root_name": bookmark.root_name,
            "date_added": created_at.isoformat() if created_at else "",
            "date_last_used": last_used_at.isoformat() if last_used_at else "",
            "guid": bookmark.guid,
            "chrome_id": bookmark.source_id,
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project="chrome_bookmarks_json",
            source_id=self._source_id(bookmark),
            source_entity_type="bookmark",
            title=bookmark.title,
            content=self._content(bookmark, folder_path),
            content_type=ContentType.ARTIFACT,
            metadata=clean_metadata(metadata),
            tags=self._tags(bookmark),
            created_at=created_at or last_used_at or now,
            updated_at=last_used_at or created_at or now,
        )

    def _source_id(self, bookmark: _Bookmark) -> str:
        if bookmark.guid:
            return digest_source_id("chrome_bookmarks_json", bookmark.guid)
        return digest_source_id("chrome_bookmarks_json", bookmark.url)

    def _folder_edge(self, bookmark: _Bookmark, unit_source_id: str) -> KnowledgeEdge | None:
        if not bookmark.folder_path:
            return None
        folder_source_id = digest_source_id("chrome_bookmarks_json_folder", bookmark.root, "/".join(bookmark.folder_path))
        edge_id = digest_source_id("chrome_bookmarks_json_edge", folder_source_id, unit_source_id)
        return KnowledgeEdge(
            id=edge_id,
            from_unit_id=folder_source_id,
            to_unit_id=unit_source_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={"folder_path": "/".join(bookmark.folder_path), "root": bookmark.root, "root_name": bookmark.root_name},
        )

    def _content(self, bookmark: _Bookmark, folder_path: str) -> str:
        parts = [bookmark.title, f"URL: {bookmark.url}", f"Root: {bookmark.root_name}"]
        if folder_path:
            parts.append(f"Folder: {folder_path}")
        return "\n".join(parts)

    def _tags(self, bookmark: _Bookmark) -> list[str]:
        tags = [bookmark.root_name]
        for index in range(len(bookmark.folder_path)):
            tags.append("/".join(bookmark.folder_path[: index + 1]))
        return list(dict.fromkeys(tag for tag in tags if tag))

    def _chrome_datetime(self, value: Any) -> datetime | None:
        text = self._text(value)
        if not text:
            return None
        try:
            micros = int(float(text))
        except ValueError:
            return None
        if micros <= 0:
            return None
        return datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=micros)

    def _root_name(self, root: str, node: dict[str, Any]) -> str:
        defaults = {
            "bookmark_bar": "Bookmarks Bar",
            "other": "Other Bookmarks",
            "synced": "Mobile Bookmarks",
            "mobile": "Mobile Bookmarks",
        }
        return defaults.get(root, self._text(node.get("name")) or root)

    def _text(self, value: Any) -> str:
        if value is None or isinstance(value, (dict, list)):
            return ""
        return str(value).strip()
