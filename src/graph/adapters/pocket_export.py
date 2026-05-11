"""Adapter for Pocket HTML exports."""

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


class _PocketLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        values = {key.lower(): value or "" for key, value in attrs}
        href = values.get("href", "").strip()
        if not href:
            self._current = None
            return
        self._current = {
            "url": href,
            "tags": values.get("tags", ""),
            "time_added": values.get("time_added", ""),
            "time_updated": values.get("time_updated", ""),
            "status": values.get("status", ""),
            "favorite": values.get("favorite", ""),
            "archive": values.get("archive", values.get("archived", "")),
        }
        self._title_parts = []

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            text = data.strip()
            if text:
                self._title_parts.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current is None:
            return
        self._current["title"] = " ".join(" ".join(self._title_parts).split())
        self.items.append(self._current)
        self._current = None
        self._title_parts = []


class PocketExportAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pocket_export"

    @property
    def entity_types(self) -> list[str]:
        return ["saved_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "saved_item" not in entity_types:
            return result

        sync_at = self._sync_datetime(since) if since else None
        for path in self._iter_paths():
            try:
                items = self._read_items(path)
            except (OSError, UnicodeDecodeError):
                continue

            for item in items:
                url = self._text(item.get("url"))
                if not url:
                    continue
                title = self._text(item.get("title")) or url
                saved_at = self._parse_datetime(self._text(item.get("time_added")))
                updated_at = self._parse_datetime(self._text(item.get("time_updated")))
                comparable_at = updated_at or saved_at
                if sync_at and comparable_at and comparable_at <= sync_at:
                    continue

                tags = self._tags(self._text(item.get("tags")))
                metadata = {
                    "url": url,
                    "title": title,
                    "tags": tags,
                    "time_added": self._text(item.get("time_added")),
                    "time_updated": self._text(item.get("time_updated")),
                    "status": self._text(item.get("status")),
                    "archived": self._archived(item),
                    "favorite": self._truthy(self._text(item.get("favorite"))),
                    "source_file": str(path),
                }
                now = datetime.now(timezone.utc)
                result.units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.POCKET_EXPORT,
                        source_id=self._source_id(url),
                        source_entity_type="saved_item",
                        title=title,
                        content=self._content(title, url, tags, metadata),
                        content_type=ContentType.ARTIFACT,
                        metadata=metadata,
                        tags=tags,
                        created_at=saved_at or updated_at or now,
                        updated_at=updated_at or saved_at or now,
                    )
                )

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _iter_paths(self) -> list[Path]:
        paths: list[Path] = []
        for raw in re.split(r"[\n,]", self.path):
            text = raw.strip()
            if not text:
                continue
            path = Path(text).expanduser()
            if path.is_dir():
                paths.extend(sorted(child for child in path.rglob("*.html") if child.is_file()))
            elif path.is_file():
                paths.append(path)
        return paths

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        parser = _PocketLinkParser()
        parser.feed(path.read_text(encoding="utf-8-sig"))
        parser.close()
        return parser.items

    def _source_id(self, url: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return f"pocket_export:{digest}"

    def _content(self, title: str, url: str, tags: list[str], metadata: dict[str, Any]) -> str:
        parts = [title, f"URL: {url}"]
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if metadata["archived"]:
            parts.append("Archived: true")
        if metadata["favorite"]:
            parts.append("Favorite: true")
        return "\n".join(parts)

    def _tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for tag in re.split(r"[,;|]", value):
            normalized = re.sub(r"\s+", " ", tag.strip().removeprefix("#")).strip().lower()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _archived(self, item: dict[str, Any]) -> bool:
        status = self._text(item.get("status")).lower()
        archive = self._text(item.get("archive"))
        return self._truthy(archive) or status in {"archive", "archived", "1"}

    def _truthy(self, value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "favorite", "favorited"}

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
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _text(self, value: object) -> str:
        return str(value or "").strip()
