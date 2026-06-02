"""Adapter for Claude conversation JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class ClaudeConversationsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "claude_conversations_json"

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
            for index, record in enumerate(_records(path)):
                unit = self._unit(record, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        conv_id = first(record, "uuid", "id", "conversation_id")
        raw_title = first(record, "title", "name")
        title = raw_title or "Claude conversation"
        messages = [item for item in record.get("messages", []) if isinstance(item, dict)] if isinstance(record.get("messages"), list) else []
        if not any([conv_id, raw_title, messages]):
            return None
        created = parse_datetime(first(record, "created_at", "createdAt"))
        updated = parse_datetime(first(record, "updated_at", "updatedAt")) or created or datetime.now(timezone.utc)
        models = sorted({model for message in messages if (model := first(message, "model"))})
        metadata = clean_metadata({"uuid": conv_id, "title": title, "created_at": created.isoformat() if created else None, "updated_at": updated.isoformat(), "message_count": len(messages), "models": models or split_single(first(record, "model")), "account": first(record, "account", "account_id"), "project": first(record, "project", "project_id"), "source_file": source_file})
        excerpt = _excerpt(messages)
        return KnowledgeUnit(source_project=self.name, source_id=f"{self.name}:{conv_id}" if conv_id else digest_source_id(self.name, title, created, excerpt, index), source_entity_type="conversation", title=title, content=excerpt or title, content_type=ContentType.ARTIFACT, metadata=metadata, tags=["claude", "conversation"], created_at=created or updated, updated_at=updated)


def _records(path: Path) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    raw = parsed
    if isinstance(parsed, dict):
        for key in ("conversations", "chats", "items"):
            if isinstance(parsed.get(key), list):
                raw = parsed[key]
                break
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])


def _excerpt(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        text = _text(message.get("content") or message.get("text"))
        if text:
            lines.append(f"{first(message, 'role', 'sender')}: {text}".strip(": "))
    return "\n".join(lines[:40])


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value if _text(item))
    if isinstance(value, dict):
        return first(value, "text", "content")
    return ""


def split_single(value: str) -> list[str]:
    return [value] if value else []
