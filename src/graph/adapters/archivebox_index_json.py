"""Adapter for ArchiveBox index JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class ArchiveBoxIndexJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "archivebox_index_json"

    @property
    def entity_types(self) -> list[str]:
        return ["archive"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "archive" not in entity_types:
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                entries = self._read_entries(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for entry in entries:
                unit = self._unit_from_entry(entry, path.name)
                if unit is None:
                    continue
                if sync_at and unit.created_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.created_at, unit.source_id)))
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

    def _read_entries(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._entries(parsed)

    def _entries(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []
        for key in ("entries", "results", "items", "snapshots", "archive"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                if self._looks_like_entry(nested):
                    return [nested]
                return [item for item in nested.values() if isinstance(item, dict)]
        if self._looks_like_entry(value):
            return [value]
        return [item for item in value.values() if isinstance(item, dict) and self._looks_like_entry(item)]

    def _looks_like_entry(self, entry: dict[str, Any]) -> bool:
        return any(key in entry for key in ("url", "base_url", "timestamp", "title"))

    def _unit_from_entry(self, entry: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        url = self._first(entry, "url", "base_url", "original_url")
        if not url:
            return None
        title = self._first(entry, "title") or url
        timestamp_text = self._first(entry, "timestamp", "bookmarked_at", "created_at", "added")
        timestamp = self._parse_datetime(timestamp_text) or datetime.now(timezone.utc)
        tags = self._tags(entry.get("tags"))
        extractor_outputs = self._extractor_outputs(entry)
        archive_paths = self._archive_paths(entry)
        status = self._first(entry, "status", "downloaded", "is_archived")

        metadata = {
            "url": url,
            "title": title,
            "timestamp": timestamp.isoformat(),
            "tags": tags,
            "status": status,
            "extractor_outputs": extractor_outputs,
            "archive_paths": archive_paths,
            "source_file": source_file,
            "entry": entry,
        }
        unit_tags = ["archivebox", *tags]
        return KnowledgeUnit(
            source_project=SourceProject.ARCHIVEBOX_INDEX_JSON,
            source_id=self._source_id(entry, url, timestamp),
            source_entity_type="archive",
            title=title,
            content=self._content(title, url, timestamp, tags, status, archive_paths),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=unit_tags,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _source_id(self, entry: dict[str, Any], url: str, timestamp: datetime) -> str:
        explicit = self._first(entry, "id", "uuid", "timestamp")
        raw = explicit or f"{url}|{timestamp.isoformat()}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"archivebox_index_json:{digest}"

    def _extractor_outputs(self, entry: dict[str, Any]) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for key in ("history", "extractors", "outputs"):
            value = entry.get(key)
            if isinstance(value, dict):
                outputs[key] = value
        return outputs

    def _archive_paths(self, entry: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for key, value in entry.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("path", "archive", "index")):
                self._append_paths(paths, value)
        return paths

    def _append_paths(self, paths: list[str], value: Any) -> None:
        if isinstance(value, str):
            if value.strip() and value.strip() not in paths:
                paths.append(value.strip())
            return
        if isinstance(value, list):
            for item in value:
                self._append_paths(paths, item)
            return
        if isinstance(value, dict):
            for item in value.values():
                self._append_paths(paths, item)

    def _tags(self, value: Any) -> list[str]:
        raw: list[Any]
        if isinstance(value, list):
            raw = value
        elif isinstance(value, str):
            raw = value.replace(";", ",").split(",")
        else:
            raw = []
        tags: list[str] = []
        for item in raw:
            tag = str(item).strip()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _content(self, title: str, url: str, timestamp: datetime, tags: list[str], status: str, archive_paths: list[str]) -> str:
        parts = [title, f"URL: {url}", f"Timestamp: {timestamp.isoformat()}"]
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if status:
            parts.append(f"Status: {status}")
        if archive_paths:
            parts.append("Archive paths: " + ", ".join(archive_paths))
        return "\n".join(parts)

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is not None and not isinstance(value, (dict, list)) and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        text = str(value).strip()
        if text.replace(".", "", 1).isdigit():
            try:
                number = float(text)
                if number > 10_000_000_000:
                    number = number / 1000
                return datetime.fromtimestamp(number, tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
