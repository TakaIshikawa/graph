"""Adapter for thread-oriented Slack JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, iter_paths
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SlackThreadsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "slack_threads_json"

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
        messages = [message for path in iter_paths(self.path, {".json"}) for message in self._messages(path)]
        for group in self._groups(messages).values():
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
        if not isinstance(raw, list):
            return []
        channel = path.stem
        return [{**item, "channel": str(item.get("channel") or channel), "source_file": str(path)} for item in raw if isinstance(item, dict)]

    def _groups(self, messages: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for message in messages:
            ts = _text(message.get("ts"))
            key = (str(message.get("channel") or ""), _text(message.get("thread_ts")) or ts)
            if ts:
                groups.setdefault(key, []).append(message)
        return groups

    def _unit(self, messages: list[dict[str, Any]]) -> KnowledgeUnit | None:
        ordered = sorted(messages, key=lambda item: _text(item.get("ts")))
        if not ordered:
            return None
        root = ordered[0]
        channel = _text(root.get("channel"))
        thread_ts = _text(root.get("thread_ts")) or _text(root.get("ts"))
        updated = _from_slack_ts(_text(ordered[-1].get("ts")))
        users = sorted({_text(item.get("user") or item.get("username")) for item in ordered if _text(item.get("user") or item.get("username"))})
        attachments = [item.get("files") or item.get("attachments") for item in ordered if item.get("files") or item.get("attachments")]
        metadata = clean_metadata({"channel": channel, "thread_ts": thread_ts, "users": users, "timestamps": [_text(item.get("ts")) for item in ordered], "permalink": _text(root.get("permalink")), "reply_count": max(0, len(ordered) - 1), "attachments": attachments, "messages": ordered})
        lines = [f"{_text(item.get('user') or item.get('username'))}: {_text(item.get('text'))}".strip(": ") for item in ordered if _text(item.get("text"))]
        return KnowledgeUnit(source_project=self.name, source_id=digest_source_id(self.name, channel, thread_ts), source_entity_type="thread", title=f"Slack thread {channel} {thread_ts}", content="\n".join(lines), content_type=ContentType.ARTIFACT, metadata=metadata, tags=["slack", "thread"], created_at=_from_slack_ts(_text(root.get("ts"))), updated_at=updated)


def _from_slack_ts(value: str) -> datetime:
    try:
        return datetime.fromtimestamp(float(value), timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()
