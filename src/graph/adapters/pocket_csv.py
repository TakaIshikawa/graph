"""Adapter for Pocket CSV exports."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class PocketCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pocket_csv"

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
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue

            for row in rows:
                url = self._first(row, "url", "given_url", "resolved_url", "item_url")
                if not url:
                    continue

                title = self._first(row, "title", "given_title", "resolved_title", "item_title") or url
                added_at = self._parse_datetime(
                    self._first(row, "time_added", "added_at", "created_at")
                )
                updated_at = self._parse_datetime(
                    self._first(row, "time_updated", "updated_at", "time_read", "time_favorited")
                )
                sync_candidate = updated_at or added_at
                if sync_at and sync_candidate and sync_candidate <= sync_at:
                    continue

                tags = self._parse_tags(self._first(row, "tags", "tag"))
                excerpt = self._first(
                    row,
                    "excerpt",
                    "resolved_excerpt",
                    "description",
                    "summary",
                )
                now = datetime.now(timezone.utc)
                result.units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.POCKET_CSV,
                        source_id=f"url:{url}",
                        source_entity_type="saved_item",
                        title=title,
                        content=self._content(title, url, excerpt, tags),
                        content_type=ContentType.ARTIFACT,
                        metadata=self._metadata(row, title, url, excerpt, tags),
                        tags=tags,
                        created_at=added_at or updated_at or now,
                        updated_at=updated_at or added_at or now,
                    )
                )

        return result

    def _iter_paths(self) -> list[Path]:
        paths: list[Path] = []
        for source in re.split(r"[\n,]", self.path):
            source = source.strip()
            if not source:
                continue
            path = Path(source).expanduser()
            if path.is_dir():
                paths.extend(sorted(child for child in path.rglob("*.csv") if child.is_file()))
            elif path.exists() and path.is_file():
                paths.append(path)
        return paths

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            rows: list[dict[str, Any]] = []
            for row in reader:
                rows.append(
                    {
                        str(key).strip(): value
                        for key, value in row.items()
                        if key is not None
                    }
                )
            return rows

    def _metadata(
        self,
        row: dict[str, Any],
        title: str,
        url: str,
        excerpt: str,
        tags: list[str],
    ) -> dict[str, Any]:
        status = self._first(row, "status", "state")
        return {
            "title": title,
            "url": url,
            "time_added": self._first(row, "time_added", "added_at", "created_at"),
            "status": status,
            "archived": self._is_archived(row, status),
            "favorite": self._is_truthy(self._first(row, "favorite", "is_favorite", "favorited")),
            "read": self._is_read(row, status),
            "excerpt": excerpt,
            "tags": tags,
        }

    def _content(self, title: str, url: str, excerpt: str, tags: list[str]) -> str:
        parts = [title, f"URL: {url}"]
        if excerpt:
            parts.append(f"Excerpt: {excerpt}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return "\n".join(parts)

    def _parse_tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for tag in re.split(r"[,;|]", value):
            normalized = re.sub(r"\s+", " ", tag.strip().removeprefix("#")).strip().lower()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _is_archived(self, row: dict[str, Any], status: str) -> bool:
        archived = self._first(row, "archived", "is_archived")
        if archived:
            return self._is_truthy(archived)
        return status.strip().lower() in {"1", "archive", "archived"}

    def _is_read(self, row: dict[str, Any], status: str) -> bool:
        read = self._first(row, "read", "is_read")
        if read:
            return self._is_truthy(read)
        status_value = status.strip().lower()
        return bool(self._first(row, "time_read")) or status_value in {
            "1",
            "archive",
            "archived",
            "read",
        }

    def _is_truthy(self, value: str) -> bool:
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
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
