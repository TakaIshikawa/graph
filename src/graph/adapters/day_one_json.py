"""Adapter for Day One JSON journal exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class DayOneJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "day_one_json"

    @property
    def entity_types(self) -> list[str]:
        return ["entry"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "entry" not in entity_types:
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
        return sorted(root.rglob("*.json"), key=lambda child: str(child))

    def _read_entries(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("entries", "journal_entries", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_entry(self, entry: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        uuid = self._first(entry, "uuid", "id")
        if not uuid:
            return None
        created_at = self._parse_datetime(self._value(entry, "creationDate", "createdDate", "created_at"))
        updated_at = self._parse_datetime(self._value(entry, "modifiedDate", "updatedDate", "updated_at"))
        if created_at is None:
            created_at = updated_at
        if created_at is None:
            return None

        content = self._first(entry, "text", "content", "body")
        tags = self._tags(entry.get("tags"))
        attachments = {
            "photos": self._attachment_metadata(entry.get("photos")),
            "audio": self._attachment_metadata(entry.get("audio") or entry.get("audios")),
        }
        metadata = {
            "uuid": uuid,
            "tags": tags,
            "location": entry.get("location") if isinstance(entry.get("location"), dict) else None,
            "weather": entry.get("weather") if isinstance(entry.get("weather"), dict) else None,
            "starred": self._parse_bool(self._value(entry, "starred", "isStarred", "favorite")),
            "attachments": attachments,
            "source_file": source_file,
        }
        return KnowledgeUnit(
            source_project=SourceProject.DAY_ONE_JSON,
            source_id=f"day_one_json:{uuid}",
            source_entity_type="entry",
            title=self._title(content, created_at),
            content=content,
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=["day_one", "journal", *tags],
            created_at=created_at,
            updated_at=updated_at or created_at,
        )

    def _title(self, content: str, created_at: datetime) -> str:
        first_line = content.strip().splitlines()[0] if content.strip() else ""
        return first_line[:80] or f"Journal entry on {created_at.date().isoformat()}"

    def _attachment_metadata(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def _tags(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(tag).strip() for tag in value if str(tag).strip()]

    def _first(self, item: dict[str, Any], *keys: str) -> str:
        value = self._value(item, *keys)
        if value is None or isinstance(value, dict | list):
            return ""
        return str(value).strip()

    def _value(self, item: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in item:
                return item[key]
        return None

    def _parse_bool(self, value: Any) -> bool | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return self._ensure_utc(value)
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return self._ensure_utc(parsed)

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
