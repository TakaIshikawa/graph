"""Adapter for Google Keep Takeout JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GoogleKeepTakeoutJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "google_keep_takeout_json"

    @property
    def entity_types(self) -> list[str]:
        return ["note"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "note" not in entity_types:
            return result
        for path in self._paths():
            try:
                record = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(record, dict):
                unit = self._unit(record, path.name)
                if unit:
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _paths(self) -> list[Path]:
        root = Path(self.path).expanduser() if self.path else None
        if root is None:
            return []
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        return sorted(root.rglob("*.json")) if root.is_dir() else []

    def _unit(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = _text(record.get("title"))
        text = _text(record.get("textContent"))
        checklist = _checklist(record.get("listContent"))
        labels = [_text(item.get("name") if isinstance(item, dict) else item) for item in record.get("labels", []) if _text(item.get("name") if isinstance(item, dict) else item)]
        created_at = _timestamp(record.get("createdTimestampUsec")) or datetime.now(timezone.utc)
        edited_at = _timestamp(record.get("userEditedTimestampUsec")) or created_at
        attachments = record.get("attachments") if isinstance(record.get("attachments"), list) else []
        if not any([title, text, checklist]):
            return None
        fallback = (text or (checklist[0]["text"] if checklist else "") or source_file).splitlines()[0][:50]
        metadata = clean_metadata({"labels": labels, "color": _text(record.get("color")), "isPinned": bool(record.get("isPinned")), "isArchived": bool(record.get("isArchived")), "createdTimestampUsec": record.get("createdTimestampUsec"), "userEditedTimestampUsec": record.get("userEditedTimestampUsec"), "checklist": checklist, "attachments": attachments, "source_file": source_file})
        return KnowledgeUnit(source_project="google_keep_takeout_json", source_id=digest_source_id("google_keep_takeout_json", source_file, title, text, record.get("createdTimestampUsec")), source_entity_type="note", title=title or fallback or "Untitled Keep note", content=_content(title, text, checklist, labels), content_type=ContentType.ARTIFACT, metadata=metadata, tags=labels, created_at=created_at, updated_at=edited_at)


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _timestamp(value: object) -> datetime | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(number / 1_000_000, tz=timezone.utc)


def _checklist(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = _text(item.get("text") or item.get("textContent"))
        if text:
            items.append({"text": text, "checked": bool(item.get("isChecked"))})
    return items


def _content(title: str, text: str, checklist: list[dict[str, object]], labels: list[str]) -> str:
    parts = [title, text]
    parts.extend(f"[{'x' if item['checked'] else ' '}] {item['text']}" for item in checklist)
    if labels:
        parts.append(f"Labels: {', '.join(labels)}")
    return "\n".join(part for part in parts if part)
