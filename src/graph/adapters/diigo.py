"""Adapter for Diigo annotated bookmark exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class DiigoAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "diigo"

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
            items = self._read_items(path)
        except (OSError, UnicodeDecodeError, csv.Error):
            return result

        sync_at = self._sync_datetime(since) if since else None
        for item in items:
            url = self._url(item)
            title = self._title(item, url)
            if not url and not title:
                continue

            created_text = self._first(item, "created_at", "created", "date", "time")
            created_at = self._parse_datetime(created_text)
            if sync_at and created_at and created_at <= sync_at:
                continue

            annotations = self._first(item, "annotations", "annotation", "comments", "comment")
            highlights = self._first(item, "highlights", "highlight")
            description = self._first(item, "description", "summary", "excerpt")
            tags = self._parse_tags(self._first(item, "tags", "tag"))
            privacy = self._first(item, "privacy", "shared", "private")
            diigo_id = self._first(item, "id", "bookmark_id")

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.DIIGO,
                    source_id=self._source_id(diigo_id, url, title),
                    source_entity_type="bookmark",
                    title=title or url or "Untitled Diigo bookmark",
                    content=self._content(title, url, description, annotations, highlights),
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "url": url,
                        "description": description,
                        "annotations": annotations,
                        "highlights": highlights,
                        "privacy": privacy,
                        "tags": tags,
                        "created_at": created_text,
                        "diigo_id": diigo_id,
                    },
                    tags=tags,
                    created_at=created_at or datetime.now(timezone.utc),
                    updated_at=created_at or datetime.now(timezone.utc),
                )
            )

        return result

    def _read_items(self, path: Path) -> list[dict[str, Any]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [
                {str(key).strip(): value for key, value in row.items() if key is not None}
                for row in csv.DictReader(handle)
            ]

    def _title(self, item: dict[str, Any], url: str) -> str:
        return self._first(item, "title", "bookmark_title", "name") or url

    def _url(self, item: dict[str, Any]) -> str:
        return self._first(item, "url", "href", "link")

    def _source_id(self, diigo_id: str, url: str, title: str) -> str:
        if diigo_id:
            return f"diigo:{diigo_id}"
        if url:
            return f"url:{url}"
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()
        return f"diigo:{digest[:24]}"

    def _content(
        self,
        title: str,
        url: str,
        description: str,
        annotations: str,
        highlights: str,
    ) -> str:
        parts = []
        if title:
            parts.append(title)
        if url:
            parts.append(f"URL: {url}")
        if description:
            parts.append(f"Description: {description}")
        if annotations:
            parts.append(f"Annotations: {annotations}")
        if highlights:
            parts.append(f"Highlights: {highlights}")
        return "\n".join(parts)

    def _parse_tags(self, value: str) -> list[str]:
        if not value:
            return []
        tags: list[str] = []
        for tag in re.split(r"[,;|]", value):
            normalized = re.sub(r"\s+", " ", tag.strip().removeprefix("#")).strip().lower()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

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
