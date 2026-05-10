"""Adapter for Simplenote JSON export files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class SimplenoteExportAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "simplenote_export"

    @property
    def entity_types(self) -> list[str]:
        return ["note"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "note" not in entity_types:
            return result

        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.exists() or not path.is_file():
            return result

        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return result

        items = self._extract_items(data)
        sync_at = self._sync_datetime(since) if since else None

        for item in items:
            if not isinstance(item, dict):
                continue

            content = self._get_text(item, "content", "text", "body")
            if not content:
                continue

            # Extract title from first line
            lines = content.splitlines()
            title = lines[0] if lines else "Untitled Simplenote"

            # Parse timestamps
            created_text = self._get_text(item, "created", "creationDate", "createdate")
            modified_text = self._get_text(item, "lastModified", "modificationDate", "modifydate")
            created_at = self._parse_timestamp(created_text)
            updated_at = self._parse_timestamp(modified_text) or created_at

            if sync_at and updated_at <= sync_at:
                continue

            # Extract tags
            tags = self._extract_tags(item.get("tags"))

            # Extract ID
            note_id = self._get_text(item, "id", "key", "simperiumKey")
            source_id = f"simplenote_export:note:{note_id}" if note_id else f"simplenote_export:note:{hash(content)}"

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.SIMPLENOTE_EXPORT,
                    source_id=source_id,
                    source_entity_type="note",
                    title=title[:100],
                    content=content,
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "note_id": note_id,
                        "tags": tags,
                        "markdown": item.get("markdown", item.get("systemTags", [])),
                    },
                    tags=tags,
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )

        result.units.sort(key=lambda u: (u.created_at, u.source_id))
        return result

    def _extract_items(self, data: Any) -> list[dict[str, Any]]:
        """Extract note items from JSON structure."""
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("activeNotes", "trashedNotes", "notes", "items"):
                items = data.get(key)
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
        return [data] if isinstance(data, dict) else []

    def _get_text(self, item: dict[str, Any], *keys: str) -> str:
        """Get first non-empty text value from keys."""
        for key in keys:
            value = item.get(key)
            if value and not isinstance(value, (dict, list)):
                return str(value).strip()
        return ""

    def _extract_tags(self, value: Any) -> list[str]:
        """Extract tags from various formats."""
        if not value:
            return []
        if isinstance(value, list):
            return [str(t).strip().lower() for t in value if t]
        if isinstance(value, str):
            return [t.strip().lower() for t in value.split(",") if t.strip()]
        return []

    def _parse_timestamp(self, value: str) -> datetime:
        """Parse timestamp in various formats."""
        if not value:
            return datetime.now(timezone.utc)

        # Try Unix timestamp
        if value.isdigit():
            try:
                return datetime.fromtimestamp(int(value), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                pass

        # Try ISO format
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return datetime.now(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        """Convert SyncState to datetime."""
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
