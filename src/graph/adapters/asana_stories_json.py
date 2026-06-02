"""Adapter for Asana task story JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class AsanaStoriesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "asana_stories_json"

    @property
    def entity_types(self) -> list[str]:
        return ["story"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "story" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._records(json.loads(path.read_text(encoding="utf-8-sig")))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records, start=1):
                record["_source_file"] = path.name
                record["_source_row"] = index
                unit = self._unit(record)
                if unit is None or (sync_at and unit.updated_at <= sync_at):
                    continue
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("stories", "data", "nodes"):
                nested = value.get(key)
                records = self._records(nested)
                if records:
                    return records
            return [value]
        return []

    def _unit(self, record: dict[str, Any]) -> KnowledgeUnit | None:
        story_id = self._text(self._get(record, "gid", "id", "story_gid"))
        task = self._dict(record.get("task"))
        task_gid = self._text(self._get(record, "task_gid", "taskGid")) or self._text(task.get("gid") or task.get("id"))
        task_name = self._text(self._get(record, "task_name", "taskName")) or self._text(task.get("name"))
        creator = self._person(record.get("created_by") or record.get("createdBy") or record.get("creator"))
        text = self._text(self._get(record, "text", "html_text", "htmlText", "body"))
        subtype = self._text(self._get(record, "resource_subtype", "resourceSubtype", "type"))
        created = parse_datetime(self._get(record, "created_at", "createdAt"))
        project = self._text(self._get(record, "project", "project_name", "projectName"))
        if not any([story_id, task_gid, task_name, text, subtype]):
            return None
        url = self._text(self._get(record, "permalink_url", "permalinkUrl", "url"))
        metadata = clean_metadata({"story_gid": story_id, "task_gid": task_gid, "task_name": task_name, "project": project, "creator": creator, "created_at": created.isoformat() if created else self._text(self._get(record, "created_at", "createdAt")), "type": self._text(self._get(record, "type")), "resource_subtype": subtype, "text": text, "liked": self._bool(self._get(record, "liked", "is_liked", "isLiked")), "permalink": url, "source_url": url, "external_url": url, "source_file": record.get("_source_file"), "source_row": record.get("_source_row")})
        now = datetime.now(timezone.utc)
        title = f"Asana story on {task_name}" if task_name else "Asana story"
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{story_id}" if story_id else digest_source_id(self.name, task_gid, text, created), source_entity_type="story", title=title, content="\n".join(part for part in [title, text or subtype, f"Task: {task_name}" if task_name else "", f"Creator: {creator}" if creator else ""] if part), content_type=ContentType.METADATA, metadata=metadata, tags=list(dict.fromkeys(tag for tag in ["asana", "story", subtype, project] if tag)), created_at=created or now, updated_at=created or now)

    def _get(self, record: dict[str, Any], *keys: str) -> Any:
        compact = {"".join(ch for ch in str(k).casefold() if ch.isalnum()): v for k, v in record.items()}
        for key in keys:
            if key in record:
                return record[key]
            value = compact.get("".join(ch for ch in key.casefold() if ch.isalnum()))
            if value is not None:
                return value
        return None

    def _dict(self, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _person(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("name") or value.get("email") or value.get("gid"))
        return self._text(value)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()

    def _bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        text = self._text(value).casefold()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
        return None
