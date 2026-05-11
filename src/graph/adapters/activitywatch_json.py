"""Adapter for ActivityWatch JSON bucket exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class ActivityWatchJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "activitywatch_json"

    @property
    def entity_types(self) -> list[str]:
        return ["activity"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "activity" not in entity_types:
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                buckets = self._read_buckets(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for bucket_id, bucket in buckets:
                for event in bucket.get("events", []):
                    if not isinstance(event, dict):
                        continue
                    unit = self._unit_from_event(bucket_id, bucket, event, path.name)
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

    def _read_buckets(self, path: Path) -> list[tuple[str, dict[str, Any]]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, dict) and isinstance(parsed.get("buckets"), dict):
            parsed = parsed["buckets"]
        if isinstance(parsed, dict) and isinstance(parsed.get("events"), list):
            return [(str(parsed.get("id") or path.stem), parsed)]
        if isinstance(parsed, dict):
            return [(str(bucket.get("id") or bucket_id), bucket) for bucket_id, bucket in parsed.items() if isinstance(bucket, dict)]
        if isinstance(parsed, list):
            return [(str(bucket.get("id") or path.stem), bucket) for bucket in parsed if isinstance(bucket, dict)]
        return []

    def _unit_from_event(
        self,
        bucket_id: str,
        bucket: dict[str, Any],
        event: dict[str, Any],
        source_file: str,
    ) -> KnowledgeUnit | None:
        timestamp = self._parse_datetime(event.get("timestamp"))
        if timestamp is None:
            return None
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        bucket_type = str(bucket.get("type") or bucket.get("bucket_type") or "").strip()
        app = self._first(data, "app", "application")
        title_text = self._first(data, "title", "status")
        url = self._first(data, "url")
        title = self._title(bucket_type, app, title_text, url)
        duration = self._parse_float(event.get("duration"))
        metadata = {
            "bucket_id": bucket_id,
            "bucket_type": bucket_type,
            "timestamp": timestamp.isoformat(),
            "duration": duration,
            "app": app,
            "title": title_text,
            "url": url,
            "data": data,
            "source_file": source_file,
        }
        tags = ["activitywatch"]
        if bucket_type:
            tags.append(bucket_type)
        if app:
            tags.append(app)
        return KnowledgeUnit(
            source_project=SourceProject.ACTIVITYWATCH_JSON,
            source_id=self._source_id(bucket_id, timestamp, data),
            source_entity_type="activity",
            title=title,
            content=self._content(bucket_type, app, title_text, url, duration),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=tags,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _source_id(self, bucket_id: str, timestamp: datetime, data: dict[str, Any]) -> str:
        data_hash = hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
        raw = f"{bucket_id}|{timestamp.isoformat()}|{data_hash}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"activitywatch_json:{digest}"

    def _title(self, bucket_type: str, app: str, title: str, url: str) -> str:
        if app and title:
            return f"{app}: {title}"
        return title or app or url or bucket_type or "ActivityWatch event"

    def _content(self, bucket_type: str, app: str, title: str, url: str, duration: float | None) -> str:
        parts = []
        if bucket_type:
            parts.append(f"Bucket type: {bucket_type}")
        if app:
            parts.append(f"App: {app}")
        if title:
            parts.append(f"Title: {title}")
        if url:
            parts.append(f"URL: {url}")
        if duration is not None:
            parts.append(f"Duration: {duration}")
        return "\n".join(parts)

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _parse_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")))
        except ValueError:
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
