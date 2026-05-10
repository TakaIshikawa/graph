"""Adapter for Pinboard HTML bookmark exports."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class _PinboardHtmlParser(HTMLParser):
    """Parse Pinboard HTML export format with DT/A tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.bookmarks: list[dict[str, Any]] = []
        self._current_link: dict[str, str] | None = None
        self._link_parts: list[str] = []
        self._pending_bookmark: dict[str, Any] | None = None
        self._capturing_description = False
        self._description_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {key.lower(): value or "" for key, value in attrs}

        if tag == "dt":
            self._finalize_description()
            return

        if tag == "a":
            if self._pending_bookmark is not None:
                self.bookmarks.append(self._pending_bookmark)
                self._pending_bookmark = None
            self._current_link = attrs_dict
            self._link_parts = []
            return

        if tag == "dd":
            if self._pending_bookmark is not None:
                self._capturing_description = True
                self._description_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "a" and self._current_link is not None:
            url = self._current_link.get("href", "").strip()
            if url:
                title = self._clean_text(" ".join(self._link_parts)) or url
                self._pending_bookmark = {
                    "url": url,
                    "title": title,
                    "add_date": self._current_link.get("add_date", "").strip(),
                    "private": self._current_link.get("private", "").strip(),
                    "toread": self._current_link.get("toread", "").strip(),
                    "tags": self._current_link.get("tags", "").strip(),
                    "description": "",
                }
            self._current_link = None
            self._link_parts = []
            return

        if tag == "dd" and self._capturing_description:
            description = self._clean_text(" ".join(self._description_parts))
            if self._pending_bookmark is not None:
                self._pending_bookmark["description"] = description
            self._capturing_description = False
            self._description_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_link is not None:
            self._link_parts.append(data)
        elif self._capturing_description:
            self._description_parts.append(data)

    def close(self) -> None:
        self._finalize_description()
        if self._pending_bookmark is not None:
            self.bookmarks.append(self._pending_bookmark)
            self._pending_bookmark = None
        super().close()

    def _clean_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def _finalize_description(self) -> None:
        if self._capturing_description:
            description = self._clean_text(" ".join(self._description_parts))
            if self._pending_bookmark is not None and description:
                self._pending_bookmark["description"] = description
            self._capturing_description = False
            self._description_parts = []


class PinboardHtmlExportAdapter(SourceAdapter):
    """Import Pinboard HTML bookmark exports."""

    @property
    def name(self) -> str:
        return "pinboard_html_export"

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
        if path is None or not path.exists():
            return result

        files = self._html_paths(path)
        sync_at = self._sync_datetime(since) if since else None

        for file_path in files:
            try:
                html = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            parser = _PinboardHtmlParser()
            parser.feed(html)
            parser.close()

            for bookmark in parser.bookmarks:
                unit = self._unit_from_bookmark(bookmark)
                if unit is None:
                    continue
                if sync_at and unit.created_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda u: u.source_id)
        return result

    def _html_paths(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root]
        return sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in (".html", ".htm")
        )

    def _unit_from_bookmark(self, bookmark: dict[str, Any]) -> KnowledgeUnit | None:
        url = bookmark.get("url", "")
        title = bookmark.get("title", "") or url
        if not url:
            return None

        description = bookmark.get("description", "")
        tags = self._parse_tags(bookmark.get("tags", ""))
        created_at = self._parse_unix_datetime(bookmark.get("add_date", ""))
        private = bookmark.get("private", "")
        toread = bookmark.get("toread", "")

        metadata: dict[str, Any] = {"url": url}
        if description:
            metadata["description"] = description
        if private:
            metadata["private"] = private
        if toread:
            metadata["toread"] = toread
        if tags:
            metadata["tags"] = tags
        if bookmark.get("add_date"):
            metadata["add_date"] = bookmark["add_date"]

        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        source_id = f"pinboard_html:{digest}"

        return KnowledgeUnit(
            source_project=SourceProject.PINBOARD,
            source_id=source_id,
            source_entity_type="bookmark",
            title=title,
            content=self._content(title, url, description, tags),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at or datetime.now(timezone.utc),
            updated_at=created_at or datetime.now(timezone.utc),
        )

    def _content(self, title: str, url: str, description: str, tags: list[str]) -> str:
        parts = [title]
        if url:
            parts.append(f"URL: {url}")
        if description:
            parts.append(description)
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _parse_tags(self, value: str) -> list[str]:
        if not value:
            return []
        raw = [t.strip().lower() for t in value.split(",")]
        return [t for t in raw if t]

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
