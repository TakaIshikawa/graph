"""Adapter for Google Search history JSON exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GoogleSearchHistoryJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_search_history_json"

    @property
    def entity_types(self) -> list[str]:
        return ["search_query"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "search_query" not in set(entity_types or self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
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
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
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

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("activity", "activities", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = self._text(record.get("title") or record.get("header"))
        query = self._query(record, title)
        url = self._url(record.get("url") or record.get("titleUrl") or record.get("link"))
        event_at = self._parse_datetime(record.get("time") or record.get("timestamp")) or self._parse_time_usec(record.get("time_usec"))
        products = self._list(record.get("products"))
        details = self._details(record.get("details"))
        device = self._text(record.get("device") or record.get("deviceName"))
        application = self._text(record.get("application") or record.get("app") or record.get("source"))
        if not query and not title and not url:
            return None
        metadata = {
            "title": title,
            "query": query,
            "url": url,
            "products": products,
            "details": details,
            "device": device,
            "application": application,
            "time": event_at.isoformat() if event_at else self._text(record.get("time")),
            "time_usec": self._text(record.get("time_usec")),
            "source_file": source_file,
            "record": record,
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.GOOGLE_SEARCH_HISTORY_JSON,
            source_id=self._source_id(query, title, url, event_at),
            source_entity_type="search_query",
            title=query or title or url,
            content=self._content(query, title, url, products),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=["google", "search_query"],
            created_at=event_at or now,
            updated_at=event_at or now,
        )

    def _query(self, record: dict[str, Any], title: str) -> str:
        explicit = self._text(record.get("query") or record.get("search_query") or record.get("searchTerm"))
        if explicit:
            return explicit
        for pattern in (r"^Searched for\s+(.+)$", r"^Search for\s+(.+)$"):
            match = re.match(pattern, title, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"')
        return ""

    def _content(self, query: str, title: str, url: str, products: list[str]) -> str:
        parts = [query or title, f"Products: {', '.join(products)}" if products else "", f"URL: {url}" if url else ""]
        return "\n".join(part for part in parts if part)

    def _source_id(self, query: str, title: str, url: str, event_at: datetime | None) -> str:
        raw = "|".join([query or title, url, event_at.isoformat() if event_at else ""])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"google_search_history_json:{digest}"

    def _details(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [self._text(item.get("name") if isinstance(item, dict) else item) for item in value if self._text(item.get("name") if isinstance(item, dict) else item)]
        text = self._text(value)
        return [text] if text else []

    def _list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [self._text(item) for item in value if self._text(item)]
        text = self._text(value)
        return [text] if text else []

    def _url(self, value: Any) -> str:
        text = self._text(value)
        return text.replace("\\u003d", "=").replace("\\u0026", "&")

    def _parse_time_usec(self, value: Any) -> datetime | None:
        text = self._text(value)
        if not text:
            return None
        try:
            return datetime.fromtimestamp(int(text) / 1_000_000, tz=timezone.utc)
        except (ValueError, OSError):
            return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = self._text(value)
        if not text:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
