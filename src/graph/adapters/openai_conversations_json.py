"""Adapter for ChatGPT/OpenAI conversation JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class OpenAIConversationsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "openai_conversations_json"

    @property
    def entity_types(self) -> list[str]:
        return ["conversation"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "conversation" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            for index, record in enumerate(_records(path, ("conversations", "items", "data"))):
                unit = self._unit(record, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        conv_id = first(record, "id", "conversation_id", "conversationId")
        title = first(record, "title", "name") or "Untitled conversation"
        messages = _messages(record)
        if not any([conv_id, title, messages]):
            return None
        created = _time(record.get("create_time")) or parse_datetime(first(record, "created_at", "createdAt"))
        updated = _time(record.get("update_time")) or parse_datetime(first(record, "updated_at", "updatedAt")) or created or datetime.now(timezone.utc)
        models = sorted({model for message in messages if (model := first(message, "model", "model_slug", "modelSlug"))})
        excerpt = _excerpt(messages)
        metadata = clean_metadata({"conversation_id": conv_id, "title": title, "create_time": created.isoformat() if created else None, "update_time": updated.isoformat(), "message_count": len(messages), "models": models, "source_file": source_file})
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{conv_id}" if conv_id else digest_source_id(self.name, title, created, excerpt, index), source_entity_type="conversation", title=title, content=excerpt or title, content_type=ContentType.ARTIFACT, metadata=metadata, tags=["openai", "conversation"], created_at=created or updated, updated_at=updated)


def _records(path: Path, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    raw = parsed
    if isinstance(parsed, dict):
        for key in keys:
            if isinstance(parsed.get(key), list):
                raw = parsed[key]
                break
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])


def _messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("messages")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    mapping = record.get("mapping")
    if isinstance(mapping, dict):
        items = [item.get("message") for item in mapping.values() if isinstance(item, dict)]
        return [item for item in items if isinstance(item, dict)]
    return []


def _excerpt(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = first(message, "role", "author_role") or first(message.get("author", {}) if isinstance(message.get("author"), dict) else {}, "role")
        text = _text(message.get("content"))
        if text:
            lines.append(f"{role}: {text}".strip(": "))
    return "\n".join(lines[:40])


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = value.get("parts")
        if isinstance(parts, list):
            return "\n".join(str(part).strip() for part in parts if str(part).strip())
        return first(value, "text", "content")
    return ""


def _time(value: Any) -> datetime | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    return parse_datetime(value)
