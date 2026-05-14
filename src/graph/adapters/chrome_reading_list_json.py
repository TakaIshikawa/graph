"""Adapter for Chrome Reading List JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class ChromeReadingListJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "chrome_reading_list_json"

    @property
    def entity_types(self) -> list[str]:
        return ["reading_list_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else set(self.entity_types)
        if "reading_list_item" not in allowed:
            return result

        sync_at = ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for record in records:
                unit = self._unit_from_record(record, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.updated_at, unit.source_id)))
        return result

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._records(parsed)

    def _records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []
        for key in ("readingList", "reading_list", "items", "entries", "urls", "data", "children"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                records = self._records(nested)
                if records:
                    return records
        return [value] if self._looks_like_record(value) else []

    def _looks_like_record(self, value: dict[str, Any]) -> bool:
        return bool(first(value, "url", "URL", "title", "name"))

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        url = first(record, "url", "URL", "link", "href")
        title = first(record, "title", "name")
        if not url and not title:
            return None

        normalized_url = self._normalize_url(url)
        added_at = self._record_datetime(record, "added_at", "addedAt", "date_added", "dateAdded", "time_added", "timeAdded")
        updated_at = self._record_datetime(record, "updated_at", "updatedAt", "last_updated", "lastUpdated", "date_modified", "dateModified")
        event_at = updated_at or added_at or datetime.now(timezone.utc)
        read = self._parse_bool(record.get("read") if "read" in record else record.get("isRead", record.get("readStatus")))
        status = first(record, "status", "read_status", "readStatus")
        domain = urlparse(normalized_url or url).netloc.lower()
        metadata = clean_metadata(
            {
                "url": url,
                "normalized_url": normalized_url,
                "domain": domain,
                "status": status,
                "read": read,
                "added_at": added_at.isoformat() if added_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
                "source_file": source_file,
                "record": dict(record),
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.CHROME_READING_LIST_JSON,
            source_id=digest_source_id("chrome_reading_list_json", normalized_url or url or title),
            source_entity_type="reading_list_item",
            title=title or url,
            content=self._content(title, url, status, read),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["chrome", "reading_list_item"],
            created_at=added_at or event_at,
            updated_at=event_at,
        )

    def _record_datetime(self, record: dict[str, Any], *keys: str) -> datetime | None:
        for key in keys:
            value = record.get(key)
            parsed = self._parse_datetime_value(value)
            if parsed is not None:
                return parsed
        return None

    def _parse_datetime_value(self, value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return self._timestamp(value)
        text = str(value).strip()
        if text.isdigit():
            parsed = self._timestamp(float(text))
            if parsed is not None:
                return parsed
        return parse_datetime(text)

    def _timestamp(self, value: float) -> datetime | None:
        try:
            timestamp = float(value)
            if timestamp > 10_000_000_000_000:
                timestamp = (timestamp / 1_000_000) - 11644473600
            elif timestamp > 10_000_000_000:
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    def _parse_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if value in (None, ""):
            return None
        text = str(value).strip().casefold()
        if text in {"true", "yes", "y", "1", "read"}:
            return True
        if text in {"false", "no", "n", "0", "unread"}:
            return False
        return None

    def _normalize_url(self, url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url.strip())
        if not parsed.scheme or not parsed.netloc:
            return url.strip()
        path = parsed.path.rstrip("/") or "/"
        return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", parsed.query, ""))

    def _content(self, title: str, url: str, status: str, read: bool | None) -> str:
        parts = [title] if title else []
        if url:
            parts.append(f"URL: {url}")
        if status:
            parts.append(f"Status: {status}")
        elif read is not None:
            parts.append(f"Read: {read}")
        return "\n".join(parts)
