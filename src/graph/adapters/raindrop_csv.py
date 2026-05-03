"""Adapter for Raindrop.io CSV bookmark exports."""

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


class RaindropCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "raindrop_csv"

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

        sync_at = self._sync_datetime(since) if since else None
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue

            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                comparable_at = unit.updated_at or unit.created_at
                if sync_at and comparable_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: unit.source_id)
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
                        self._normalize_key(str(key)): value
                        for key, value in row.items()
                        if key is not None
                    }
                )
            return rows

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        url = self._first(row, "url", "link", "href")
        title = self._first(row, "title", "name") or url
        excerpt = self._first(row, "excerpt", "description", "summary")
        note = self._first(row, "note", "notes")
        if not url and not title and not excerpt and not note:
            return None

        folder = self._first(row, "folder", "collection", "collection_title")
        domain = self._first(row, "domain", "host")
        tags = self._parse_tags(self._first(row, "tags", "tag"))
        created_text = self._first(row, "created", "created_at", "createdat", "date")
        updated_text = self._first(row, "updated", "updated_at", "last_update", "lastupdate")
        created_at = self._parse_datetime(created_text)
        updated_at = self._parse_datetime(updated_text)
        now = datetime.now(timezone.utc)

        return KnowledgeUnit(
            source_project=SourceProject.RAINDROP_CSV,
            source_id=self._source_id(url, title, excerpt, note),
            source_entity_type="bookmark",
            title=title or "Untitled Raindrop bookmark",
            content=self._content(title, url, excerpt, note, folder, domain, tags),
            content_type=ContentType.ARTIFACT,
            metadata={
                "title": title,
                "url": url,
                "excerpt": excerpt,
                "note": note,
                "folder": folder,
                "created_at": created_text,
                "updated_at": updated_text,
                "domain": domain,
                "tags": tags,
                "source_file": source_file,
            },
            tags=tags,
            created_at=created_at or updated_at or now,
            updated_at=updated_at or created_at or now,
        )

    def _source_id(self, url: str, title: str, excerpt: str, note: str) -> str:
        if url:
            return f"url:{url}"
        digest = hashlib.sha256(f"{title}\n{excerpt}\n{note}".encode("utf-8")).hexdigest()
        return f"raindrop_csv:{digest[:24]}"

    def _content(
        self,
        title: str,
        url: str,
        excerpt: str,
        note: str,
        folder: str,
        domain: str,
        tags: list[str],
    ) -> str:
        parts = []
        if title:
            parts.append(title)
        if url:
            parts.append(f"URL: {url}")
        if excerpt:
            parts.append(f"Excerpt: {excerpt}")
        if note:
            parts.append(f"Note: {note}")
        if folder:
            parts.append(f"Folder: {folder}")
        if domain:
            parts.append(f"Domain: {domain}")
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

    def _normalize_key(self, key: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")

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
