"""Adapter for Chrome Bookmarks JSON exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

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


@dataclass
class _Folder:
    name: str
    folder_path: tuple[str, ...]
    parent_folder_path: tuple[str, ...]
    root: str
    root_name: str
    child_bookmark_count: int
    child_folder_count: int


class ChromeBookmarksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "chrome_bookmarks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["bookmark", "domain", "folder"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        bookmark_units: list[KnowledgeUnit] = []
        folder_records: list[tuple[_Folder, str]] = []
        for path in self._iter_paths():
            try:
                parsed = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            bookmarks, folders = self._bookmarks_and_folders(parsed)
            folder_records.extend((folder, path.name) for folder in folders)
            for bookmark in bookmarks:
                created_at = self._chrome_datetime(bookmark.date_added)
                last_used_at = self._chrome_datetime(bookmark.date_last_used)
                comparable_at = last_used_at or created_at
                if sync_at and comparable_at and comparable_at <= sync_at:
                    continue
                unit = self._unit(bookmark, path.name, created_at, last_used_at)
                bookmark_units.append(unit)
                if {"bookmark", "folder"}.issubset(allowed_types):
                    edge = self._folder_edge(bookmark, unit.source_id)
                    if edge:
                        result.edges.append(edge)

        domain_units = self._domain_units(bookmark_units)
        folder_units = self._folder_units(folder_records, bookmark_units)
        if "bookmark" in allowed_types:
            result.units.extend(bookmark_units)
        if "domain" in allowed_types:
            result.units.extend(domain_units)
        if "folder" in allowed_types:
            result.units.extend(folder_units)
        if {"bookmark", "domain"}.issubset(allowed_types):
            result.edges.extend(self._domain_edges(domain_units, bookmark_units))
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

    def _bookmarks_and_folders(self, parsed: Any) -> tuple[list[_Bookmark], list[_Folder]]:
        if not isinstance(parsed, dict):
            return [], []
        roots = parsed.get("roots")
        if not isinstance(roots, dict):
            roots = parsed
        bookmarks: list[_Bookmark] = []
        folders: list[_Folder] = []
        for root_key, node in roots.items():
            if not isinstance(node, dict):
                continue
            root_name = self._root_name(str(root_key), node)
            self._walk(node, (), (), str(root_key), root_name, bookmarks, folders)
        return bookmarks, folders

    def _walk(
        self,
        node: dict[str, Any],
        folder_path: tuple[str, ...],
        parent_folder_path: tuple[str, ...],
        root: str,
        root_name: str,
        bookmarks: list[_Bookmark],
        folders: list[_Folder],
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
            folders.append(
                _Folder(
                    name=name or root_name,
                    folder_path=next_path,
                    parent_folder_path=parent_folder_path,
                    root=root,
                    root_name=root_name,
                    child_bookmark_count=sum(1 for child in children if isinstance(child, dict) and (child.get("type") == "url" or child.get("url"))),
                    child_folder_count=sum(1 for child in children if isinstance(child, dict) and not (child.get("type") == "url" or child.get("url"))),
                )
            )
            for child in children:
                if isinstance(child, dict):
                    self._walk(child, next_path, next_path, root, root_name, bookmarks, folders)

    def _unit(
        self,
        bookmark: _Bookmark,
        source_file: str,
        created_at: datetime | None,
        last_used_at: datetime | None,
    ) -> KnowledgeUnit:
        now = datetime.now(timezone.utc)
        folder_path = "/".join(bookmark.folder_path)
        url_metadata = self._url_metadata(bookmark.url)
        metadata = {
            "url": bookmark.url,
            "domain": self._domain(bookmark.url),
            **url_metadata,
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

    def _domain_units(self, bookmarks: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for bookmark in bookmarks:
            domain = str(bookmark.metadata.get("domain") or "").strip()
            if domain:
                grouped.setdefault(domain, []).append(bookmark)

        units: list[KnowledgeUnit] = []
        now = datetime.now(timezone.utc)
        for domain, domain_bookmarks in sorted(grouped.items()):
            source_ids = sorted(bookmark.source_id for bookmark in domain_bookmarks)
            roots = sorted({str(bookmark.metadata.get("root")) for bookmark in domain_bookmarks if bookmark.metadata.get("root")})
            folder_paths = sorted({str(bookmark.metadata.get("folder_path")) for bookmark in domain_bookmarks if bookmark.metadata.get("folder_path")})
            added = [
                parsed
                for bookmark in domain_bookmarks
                if (parsed := self._iso_datetime(bookmark.metadata.get("date_added"))) is not None
            ]
            last_used = [
                parsed
                for bookmark in domain_bookmarks
                if (parsed := self._iso_datetime(bookmark.metadata.get("date_last_used"))) is not None
            ]
            created_at = min(added, default=min((bookmark.created_at for bookmark in domain_bookmarks), default=now))
            updated_at = max(last_used or added, default=max((bookmark.updated_at for bookmark in domain_bookmarks), default=now))
            units.append(
                KnowledgeUnit(
                    source_project="chrome_bookmarks_json",
                    source_id=self._domain_source_id(domain),
                    source_entity_type="domain",
                    title=domain,
                    content=f"Chrome bookmark domain: {domain}\nBookmarks: {len(domain_bookmarks)}",
                    content_type=ContentType.METADATA,
                    metadata=clean_metadata(
                        {
                            "domain": domain,
                            "bookmark_count": len(domain_bookmarks),
                            "roots": roots,
                            "folder_paths": folder_paths,
                            "bookmark_source_ids": source_ids,
                            "first_added_at": min(added).isoformat() if added else "",
                            "last_used_at": max(last_used).isoformat() if last_used else "",
                        }
                    ),
                    tags=["chrome-bookmark-domain", domain],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _folder_units(self, folders: list[tuple[_Folder, str]], bookmarks: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        units: dict[str, KnowledgeUnit] = {}
        now = datetime.now(timezone.utc)
        for folder, source_file in folders:
            folder_path = "/".join(folder.folder_path)
            parent_path = "/".join(folder.parent_folder_path)
            descendant_bookmarks = [
                bookmark
                for bookmark in bookmarks
                if bookmark.metadata.get("root") == folder.root
                and (
                    str(bookmark.metadata.get("folder_path") or "") == folder_path
                    or (folder_path and str(bookmark.metadata.get("folder_path") or "").startswith(f"{folder_path}/"))
                    or (not folder_path and bookmark.metadata.get("root") == folder.root)
                )
            ]
            added = [
                parsed
                for bookmark in descendant_bookmarks
                if (parsed := self._iso_datetime(bookmark.metadata.get("date_added"))) is not None
            ]
            metadata = {
                "root": folder.root,
                "root_name": folder.root_name,
                "folder_path": folder_path,
                "parent_folder_path": parent_path,
                "child_bookmark_count": folder.child_bookmark_count,
                "child_folder_count": folder.child_folder_count,
                "bookmark_count": len(descendant_bookmarks),
                "source_files": [source_file],
                "first_added_at": min(added).isoformat() if added else "",
                "latest_added_at": max(added).isoformat() if added else "",
            }
            unit = KnowledgeUnit(
                source_project="chrome_bookmarks_json",
                source_id=self._folder_source_id(folder.root, folder_path),
                source_entity_type="folder",
                title=folder_path or folder.root_name,
                content=f"Chrome bookmark folder: {folder_path or folder.root_name}",
                content_type=ContentType.METADATA,
                metadata=metadata,
                tags=list(dict.fromkeys(tag for tag in ["chrome-bookmark-folder", folder.root_name, folder_path] if tag)),
                created_at=min(added, default=now),
                updated_at=max(added, default=now),
            )
            existing = units.get(unit.source_id)
            if existing is None:
                units[unit.source_id] = unit
            else:
                files = sorted({*(existing.metadata.get("source_files") or []), source_file})
                existing.metadata["source_files"] = files
        return list(units.values())

    def _domain_edges(self, domains: list[KnowledgeUnit], bookmarks: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        domain_ids = {str(unit.metadata.get("domain")): unit.source_id for unit in domains}
        edges: list[KnowledgeEdge] = []
        for bookmark in bookmarks:
            domain_id = domain_ids.get(str(bookmark.metadata.get("domain") or ""))
            if not domain_id:
                continue
            edge_id = digest_source_id("chrome_bookmarks_json_domain_edge", domain_id, bookmark.source_id)
            edges.append(
                KnowledgeEdge(
                    id=edge_id,
                    from_unit_id=domain_id,
                    to_unit_id=bookmark.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={"relation_type": "domain_contains_bookmark", "domain": bookmark.metadata.get("domain")},
                )
            )
        return edges

    def _domain_source_id(self, domain: str) -> str:
        return digest_source_id("chrome_bookmarks_json_domain", domain)

    def _domain(self, url: str) -> str:
        host = urlparse(url).hostname or urlparse(f"https://{url}").hostname or ""
        return host.rstrip(".").casefold()

    def _url_metadata(self, url: str) -> dict[str, str]:
        parsed = urlparse(url)
        if not parsed.scheme and not parsed.netloc:
            parsed = urlparse(f"https://{url}")
        query_param_keys = ",".join(sorted({key for key, _value in parse_qsl(parsed.query, keep_blank_values=True)}))
        return {
            "url_scheme": parsed.scheme.casefold(),
            "url_path": parsed.path,
            "query_param_keys": query_param_keys,
            "has_fragment": "true" if parsed.fragment else "false",
        }

    def _iso_datetime(self, value: Any) -> datetime | None:
        text = self._text(value)
        if not text:
            return None
        try:
            return ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None

    def _folder_edge(self, bookmark: _Bookmark, unit_source_id: str) -> KnowledgeEdge | None:
        if not bookmark.folder_path:
            return None
        folder_source_id = self._folder_source_id(bookmark.root, "/".join(bookmark.folder_path))
        edge_id = digest_source_id("chrome_bookmarks_json_edge", folder_source_id, unit_source_id)
        return KnowledgeEdge(
            id=edge_id,
            from_unit_id=folder_source_id,
            to_unit_id=unit_source_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={"folder_path": "/".join(bookmark.folder_path), "root": bookmark.root, "root_name": bookmark.root_name},
        )

    def _folder_source_id(self, root: str, folder_path: str) -> str:
        return digest_source_id("chrome_bookmarks_json_folder", root, folder_path)

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
