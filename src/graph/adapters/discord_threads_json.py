"""Adapter for Discord channel and thread JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class DiscordThreadsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "discord_threads_json"

    @property
    def entity_types(self) -> list[str]:
        return ["thread"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "thread" not in set(entity_types or self.entity_types):
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        groups: dict[str, list[dict[str, Any]]] = {}
        for path in iter_paths(self.path, {".json"}):
            for message in self._messages(path):
                groups.setdefault(self._group_key(message), []).append(message)
        for group in groups.values():
            unit = self._unit(group)
            if unit and (sync_at is None or unit.updated_at > sync_at):
                result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _messages(self, path: Path) -> list[dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        raw = data.get("messages") if isinstance(data, dict) else data
        channel = data.get("channel") if isinstance(data, dict) else {}
        channel_id = _text(channel.get("id") if isinstance(channel, dict) else "")
        if not isinstance(raw, list):
            return []
        return [{**item, "channel_id": _text(item.get("channel_id") or channel_id), "source_file": str(path)} for item in raw if isinstance(item, dict)]

    def _group_key(self, message: dict[str, Any]) -> str:
        return _text(message.get("thread_id") or message.get("threadId") or message.get("conversation_id") or message.get("channel_id") or message.get("id"))

    def _unit(self, messages: list[dict[str, Any]]) -> KnowledgeUnit | None:
        ordered = sorted(messages, key=lambda item: _text(item.get("timestamp") or item.get("created_at")))
        root = ordered[0]
        thread_id = self._group_key(root)
        channel_id = _text(root.get("channel_id"))
        created = parse_datetime(_text(root.get("timestamp") or root.get("created_at"))) or datetime.now(timezone.utc)
        updated = parse_datetime(_text(ordered[-1].get("timestamp") or ordered[-1].get("created_at"))) or created
        authors = sorted({_author(item.get("author")) for item in ordered if _author(item.get("author"))})
        attachments = [att for item in ordered for att in (item.get("attachments") or [])]
        embed_urls = [_text(embed.get("url")) for item in ordered for embed in (item.get("embeds") or []) if isinstance(embed, dict) and _text(embed.get("url"))]
        lines = [f"{_author(item.get('author'))}: {_text(item.get('content'))}".strip(": ") for item in ordered if _text(item.get("content"))]
        metadata = clean_metadata({"thread_id": thread_id, "channel_id": channel_id, "authors": authors, "message_count": len(ordered), "attachments": attachments, "embed_urls": embed_urls, "messages": ordered})
        return KnowledgeUnit(source_project=self.name, source_id=digest_source_id(self.name, thread_id), source_entity_type="thread", title=f"Discord thread {thread_id}", content="\n".join(lines), content_type=ContentType.ARTIFACT, metadata=metadata, tags=["discord", "thread"], created_at=created, updated_at=updated)


def _author(value: Any) -> str:
    return _text(value.get("username") or value.get("name") or value.get("id")) if isinstance(value, dict) else _text(value)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()
