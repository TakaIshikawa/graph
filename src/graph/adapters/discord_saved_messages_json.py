"""Adapter for Discord saved or pinned message JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class DiscordSavedMessagesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "discord_saved_messages_json"

    @property
    def entity_types(self) -> list[str]:
        return ["message"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "message" not in set(entity_types or self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
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
        root = Path(self.path).expanduser()
        if not self.path:
            return []
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.json") if path.is_file())

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("messages", "saved_messages", "pinned_messages", "items", "data"):
                if isinstance(parsed.get(key), list):
                    records = []
                    for item in parsed[key]:
                        if isinstance(item, dict):
                            merged = dict(item)
                            for parent_key in ("guild", "guild_name", "server", "server_name", "channel", "channel_name", "channel_id"):
                                if parent_key in parsed and parent_key not in merged:
                                    merged[parent_key] = parsed[parent_key]
                            records.append(merged)
                    return records
            return [parsed]
        return []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        message = record.get("message") if isinstance(record.get("message"), dict) else record
        message_id = self._text(message.get("id") or message.get("message_id"))
        content = self._text(message.get("content") or message.get("text"))
        attachments = self._attachments(message.get("attachments"))
        embeds = self._embeds(message.get("embeds"))
        if not content and not attachments and not embeds:
            return None
        timestamp = parse_datetime(message.get("timestamp") or message.get("created_at"))
        edited = parse_datetime(message.get("edited_timestamp") or message.get("edited_at"))
        metadata = {
            "guild": self._context_text(record, message, "guild", "guild_name", "server", "server_name"),
            "channel": self._context_text(record, message, "channel", "channel_name"),
            "channel_id": self._text(message.get("channel_id") or record.get("channel_id")),
            "message_id": message_id,
            "author": self._author(message.get("author")),
            "content": content,
            "timestamp": timestamp.isoformat() if timestamp else self._text(message.get("timestamp")),
            "edited_at": edited.isoformat() if edited else self._text(message.get("edited_timestamp")),
            "attachments": attachments,
            "embeds": embeds,
            "reply_to": self._reply(message),
            "source_file": source_file,
        }
        now = datetime.now(timezone.utc)
        title = self._title(metadata)
        return KnowledgeUnit(
            source_project=SourceProject.DISCORD_SAVED_MESSAGES_JSON,
            source_id=f"discord_saved_messages_json:{message_id}" if message_id else digest_source_id("discord_saved_messages_json", metadata["channel_id"], content, timestamp),
            source_entity_type="message",
            title=title,
            content=self._content(content, metadata),
            content_type=ContentType.INSIGHT,
            metadata=clean_metadata(metadata),
            tags=list(dict.fromkeys(tag for tag in ["discord", "message", metadata["guild"], metadata["channel"]] if tag)),
            created_at=timestamp or now,
            updated_at=edited or timestamp or now,
        )

    def _context_text(self, record: dict[str, Any], message: dict[str, Any], *keys: str) -> str:
        for key in keys:
            for source in (message, record):
                value = source.get(key)
                if isinstance(value, dict):
                    value = value.get("name") or value.get("title")
                text = self._text(value)
                if text:
                    return text
        return ""

    def _author(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("username") or value.get("name") or value.get("global_name") or value.get("id"))
        return self._text(value)

    def _attachments(self, value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        return [clean_metadata({"id": self._text(item.get("id")), "filename": self._text(item.get("filename")), "url": self._text(item.get("url"))}) for item in value if isinstance(item, dict)]

    def _embeds(self, value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        return [clean_metadata({"title": self._text(item.get("title")), "description": self._text(item.get("description")), "url": self._text(item.get("url"))}) for item in value if isinstance(item, dict)]

    def _reply(self, message: dict[str, Any]) -> dict[str, str]:
        ref = message.get("message_reference") or message.get("referenced_message") or message.get("reply_to")
        if not isinstance(ref, dict):
            return {}
        return clean_metadata({"message_id": self._text(ref.get("message_id") or ref.get("id")), "channel_id": self._text(ref.get("channel_id")), "guild_id": self._text(ref.get("guild_id"))})

    def _title(self, metadata: dict[str, Any]) -> str:
        prefix = " / ".join(part for part in [metadata.get("guild"), metadata.get("channel")] if part)
        author = metadata.get("author") or "Discord message"
        return f"{prefix}: {author}" if prefix else str(author)

    def _content(self, content: str, metadata: dict[str, Any]) -> str:
        parts = [content]
        if metadata.get("author"):
            parts.append(f"Author: {metadata['author']}")
        for attachment in metadata.get("attachments") or []:
            parts.append(f"Attachment: {attachment.get('filename') or attachment.get('url')}")
        for embed in metadata.get("embeds") or []:
            parts.append(f"Embed: {embed.get('title') or embed.get('url')}")
        return "\n".join(part for part in parts if part)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
