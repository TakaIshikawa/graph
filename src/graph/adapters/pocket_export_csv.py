"""Adapter for Pocket CSV exports."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PocketExportCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pocket_export_csv"

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
                unit = self._unit_from_row(row, path)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit_from_row(self, row: dict[str, Any], path: Path) -> KnowledgeUnit | None:
        url = self._first(row, "url", "given_url", "resolved_url", "item_url")
        if not url:
            return None

        title = self._first(row, "title", "given_title", "resolved_title", "item_title") or url
        added_text = self._first(row, "time_added", "added_at", "created_at")
        added_at = self._parse_datetime(added_text)
        updated_at = self._parse_datetime(
            self._first(row, "time_updated", "updated_at", "time_read", "time_favorited")
        )
        tags = self._parse_tags(self._first(row, "tags", "tag"))
        status = self._normalize_status(self._first(row, "status", "state"))
        favorite = self._is_truthy(self._first(row, "favorite", "is_favorite", "favorited"))
        archived = self._is_archived(row, status)
        read = self._is_read(row, status)
        excerpt = self._first(row, "excerpt", "resolved_excerpt", "description", "summary")
        now = datetime.now(timezone.utc)

        return KnowledgeUnit(
            source_project="pocket_export_csv",
            source_id=self._source_id(url),
            source_entity_type="saved_item",
            title=title,
            content=self._content(title, url, status, favorite, tags, excerpt),
            content_type=ContentType.ARTIFACT,
            metadata=self._metadata(
                title=title,
                url=url,
                added_text=added_text,
                added_at=added_at,
                status=status,
                favorite=favorite,
                archived=archived,
                read=read,
                tags=tags,
                excerpt=excerpt,
                source_file=path.name,
            ),
            tags=tags,
            created_at=added_at or updated_at or now,
            updated_at=updated_at or added_at or now,
        )

    def _iter_paths(self) -> list[Path]:
        paths: list[Path] = []
        for source in re.split(r"[\n,]", self.path):
            source = source.strip()
            if not source:
                continue
            path = Path(source).expanduser()
            if path.is_dir():
                paths.extend(sorted(child for child in path.rglob("*.csv") if child.is_file()))
            elif path.is_file():
                paths.append(path)
        return paths

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            return [
                {str(key).strip(): value for key, value in row.items() if key is not None}
                for row in reader
            ]

    def _metadata(
        self,
        *,
        title: str,
        url: str,
        added_text: str,
        added_at: datetime | None,
        status: str,
        favorite: bool,
        archived: bool,
        read: bool,
        tags: list[str],
        excerpt: str,
        source_file: str,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "title": title,
            "url": url,
            "source_url": url,
            "external_url": url,
            "status": status,
            "favorite": favorite,
            "archived": archived,
            "read": read,
            "tags": tags,
            "source_file": source_file,
        }
        if added_text:
            metadata["time_added"] = added_text
        if added_at:
            metadata["added_at"] = added_at.isoformat()
        if excerpt:
            metadata["excerpt"] = excerpt
        return metadata

    def _content(
        self,
        title: str,
        url: str,
        status: str,
        favorite: bool,
        tags: list[str],
        excerpt: str,
    ) -> str:
        parts = [title, f"URL: {url}"]
        if status:
            parts.append(f"Status: {status}")
        if favorite:
            parts.append("Favorite: true")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if excerpt:
            parts.append(f"Excerpt: {excerpt}")
        return "\n".join(parts)

    def _source_id(self, url: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return f"pocket_export_csv:{digest}"

    def _parse_tags(self, value: str) -> list[str]:
        tags: list[str] = []
        for tag in re.split(r"[,;|]", value):
            normalized = re.sub(r"\s+", " ", tag.strip().removeprefix("#")).strip().lower()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _normalize_status(self, value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {"0", "active", "unread", "saved"}:
            return "active"
        if normalized == "read":
            return "read"
        if normalized in {"1", "archive", "archived"}:
            return "archived"
        if normalized in {"2", "delete", "deleted"}:
            return "deleted"
        return normalized

    def _is_archived(self, row: dict[str, Any], status: str) -> bool:
        archived = self._first(row, "archived", "is_archived")
        if archived:
            return self._is_truthy(archived)
        return status == "archived"

    def _is_read(self, row: dict[str, Any], status: str) -> bool:
        read = self._first(row, "read", "is_read")
        if read:
            return self._is_truthy(read)
        return bool(self._first(row, "time_read")) or status in {"archived", "read"}

    def _is_truthy(self, value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "favorite", "favorited"}

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is None:
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
        parsed = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
