"""Adapter for Discord message JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class DiscordJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "discord_json"

    @property
    def entity_types(self) -> list[str]:
        return ["discord_message"]

    def __init__(self, path: str = "", *, root_path: str = "") -> None:
        self.path = path or root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "discord_message" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        sync_at = self._sync_datetime(since) if since else None
        references_by_source_id: dict[str, list[dict[str, str]]] = {}

        for path in self._json_files(root):
            for index, record in enumerate(self._read_message_records(path)):
                unit = self._message_unit(record, path, root, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
                references_by_source_id[unit.source_id] = self._references(record["message"])

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        included_source_ids = {unit.source_id for unit in result.units}
        source_id_by_message_id = {
            self._message_index_key(unit.metadata.get("message_id")): unit.source_id
            for unit in result.units
            if self._message_index_key(unit.metadata.get("message_id"))
        }

        emitted_edges: set[tuple[str, str, str]] = set()
        for source_id in sorted(references_by_source_id):
            if source_id not in included_source_ids:
                continue
            for reference in references_by_source_id[source_id]:
                target_id = self._reference_source_id(reference, source_id_by_message_id)
                if not target_id or target_id == source_id or target_id not in included_source_ids:
                    continue
                referenced_message_id = reference.get("message_id", "")
                edge_key = (source_id, target_id, referenced_message_id)
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(source_id, target_id, referenced_message_id),
                        from_unit_id=source_id,
                        to_unit_id=target_id,
                        relation=EdgeRelation.REFERENCES,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.DISCORD_JSON.value,
                            "from_entity_type": "discord_message",
                            "to_entity_type": "discord_message",
                            "relation_type": "discord_reply_reference",
                            "referenced_message_id": referenced_message_id,
                            "referenced_channel_id": reference.get("channel_id", ""),
                            "referenced_server_id": reference.get("server_id", ""),
                        },
                    )
                )

        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id))
        return result

    def _json_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() == ".json" else []
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.json") if path.is_file())

    def _read_message_records(self, path: Path) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []

        context = self._context_from_path(path)
        if isinstance(parsed, dict):
            context.update(self._context_from_payload(parsed))

        messages = self._message_records(parsed)
        return [{"message": message, "context": context} for message in messages]

    def _message_records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, dict):
            return []
        for key in ("messages", "Messages", "items", "records"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        if self._message_id(value) or self._content(value):
            return [value]
        return []

    def _message_unit(
        self,
        record: dict[str, Any],
        path: Path,
        root: Path,
        index: int,
    ) -> KnowledgeUnit | None:
        message = record["message"]
        context = record["context"]
        content = self._content(message)
        attachments = self._attachments(message.get("attachments") or message.get("Attachments"))
        if not content and attachments:
            content = "\n".join(
                item for attachment in attachments for item in [attachment.get("filename") or attachment.get("url", "")]
                if item
            )
        if not content:
            return None

        message_id = self._message_id(message) or self._fallback_message_id(path, index, content)
        timestamp_text = self._first(
            message,
            "timestamp",
            "Timestamp",
            "created_at",
            "createdAt",
            "date",
        )
        edited_timestamp_text = self._first(
            message,
            "edited_timestamp",
            "editedTimestamp",
            "edited_at",
            "editedAt",
        )
        created_at = self._parse_datetime(timestamp_text) or datetime.now(timezone.utc)
        updated_at = self._parse_datetime(edited_timestamp_text) or created_at
        author = self._author(message.get("author") or message.get("Author") or message)
        channel = self._channel(message, context, path)
        server = self._server(message, context)
        source_path = self._relative_path(path, root)

        metadata = {
            "message_id": message_id,
            "channel_id": channel["id"],
            "channel_name": channel["name"],
            "server_id": server["id"],
            "server_name": server["name"],
            "author": author,
            "timestamp": timestamp_text,
            "edited_timestamp": edited_timestamp_text,
            "attachments": attachments,
            "attachment_count": len(attachments),
            "source_path": source_path,
            "path": source_path,
        }
        references = self._references(message)
        if references:
            metadata["references"] = references

        tags = ["discord"]
        if server["name"]:
            tags.append(f"discord-server-{self._tag_value(server['name'])}")
        if channel["name"]:
            tags.append(f"discord-channel-{self._tag_value(channel['name'])}")

        return KnowledgeUnit(
            source_project=SourceProject.DISCORD_JSON,
            source_id=self._source_id(channel["id"] or channel["name"], message_id),
            source_entity_type="discord_message",
            title=self._title(channel["name"], created_at, author.get("display_name") or author.get("username", "")),
            content=" ".join(content.split()),
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=tags,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _context_from_payload(self, payload: dict[str, Any]) -> dict[str, str]:
        guild = payload.get("guild") or payload.get("server") or {}
        channel = payload.get("channel") or {}
        return {
            "server_id": self._object_value(guild, "id", "guild_id", "server_id"),
            "server_name": self._object_value(guild, "name", "guild_name", "server_name"),
            "channel_id": self._object_value(channel, "id", "channel_id"),
            "channel_name": self._object_value(channel, "name", "channel_name"),
        }

    def _context_from_path(self, path: Path) -> dict[str, str]:
        context = {
            "server_id": "",
            "server_name": "",
            "channel_id": "",
            "channel_name": path.parent.name if path.parent.name else path.stem,
        }
        channel_path = path.with_name("channel.json")
        try:
            parsed = json.loads(channel_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return context
        if isinstance(parsed, dict):
            sibling_context = self._context_from_payload({"channel": parsed})
            if not sibling_context["channel_name"]:
                sibling_context["channel_name"] = self._object_value(parsed, "name", "channel_name")
            if not sibling_context["channel_id"]:
                sibling_context["channel_id"] = self._object_value(parsed, "id", "channel_id")
            context.update({key: value for key, value in sibling_context.items() if value})
        return context

    def _channel(self, message: dict[str, Any], context: dict[str, str], path: Path) -> dict[str, str]:
        channel = message.get("channel") or message.get("Channel") or {}
        channel_id = (
            self._first(message, "channel_id", "channelId", "Channel ID")
            or self._object_value(channel, "id", "channel_id")
            or context.get("channel_id", "")
        )
        channel_name = (
            self._first(message, "channel_name", "channelName", "Channel")
            or self._object_value(channel, "name", "channel_name")
            or context.get("channel_name", "")
            or path.parent.name
            or path.stem
        )
        return {"id": channel_id, "name": channel_name}

    def _server(self, message: dict[str, Any], context: dict[str, str]) -> dict[str, str]:
        server = message.get("guild") or message.get("server") or message.get("Guild") or {}
        return {
            "id": self._first(message, "guild_id", "guildId", "server_id", "serverId")
            or self._object_value(server, "id", "guild_id", "server_id")
            or context.get("server_id", ""),
            "name": self._first(message, "guild_name", "guildName", "server_name", "serverName")
            or self._object_value(server, "name", "guild_name", "server_name")
            or context.get("server_name", ""),
        }

    def _author(self, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {"id": "", "username": "", "display_name": ""}
        username = self._first(value, "username", "name", "Username", "author", "Author")
        display_name = self._first(value, "global_name", "globalName", "display_name", "displayName", "nickname")
        return {
            "id": self._first(value, "id", "user_id", "author_id", "User ID"),
            "username": username,
            "display_name": display_name or username,
            "discriminator": self._first(value, "discriminator"),
            "bot": str(value.get("bot", "")).lower() if value.get("bot") is not None else "",
        }

    def _attachments(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        attachments: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    attachments.append({"url": item.strip()})
                continue
            if not isinstance(item, dict):
                continue
            attachment = {
                "id": self._first(item, "id"),
                "filename": self._first(item, "filename", "fileName", "name"),
                "url": self._first(item, "url", "proxy_url", "proxyUrl"),
                "content_type": self._first(item, "content_type", "contentType"),
                "size": item.get("size") if isinstance(item.get("size"), int | float) else None,
            }
            attachments.append({key: value for key, value in attachment.items() if value not in {"", None}})
        return attachments

    def _references(self, message: dict[str, Any]) -> list[dict[str, str]]:
        values = [
            message.get("message_reference"),
            message.get("messageReference"),
            message.get("reference"),
            message.get("referenced_message"),
            message.get("referencedMessage"),
        ]
        references: list[dict[str, str]] = []
        for value in values:
            reference = self._reference(value)
            if reference and reference not in references:
                references.append(reference)
        return references

    def _reference(self, value: Any) -> dict[str, str] | None:
        if isinstance(value, str):
            message_id = value.strip()
            return {"message_id": message_id, "channel_id": "", "server_id": ""} if message_id else None
        if not isinstance(value, dict):
            return None
        message_id = self._first(value, "message_id", "messageId", "id")
        if not message_id:
            return None
        return {
            "message_id": message_id,
            "channel_id": self._first(value, "channel_id", "channelId"),
            "server_id": self._first(value, "guild_id", "guildId", "server_id", "serverId"),
        }

    def _reference_source_id(
        self,
        reference: dict[str, str],
        source_id_by_message_id: dict[str, str],
    ) -> str:
        return source_id_by_message_id.get(self._message_index_key(reference.get("message_id")), "")

    def _source_id(self, channel: str, message_id: str) -> str:
        channel_part = channel or "unknown-channel"
        return f"discord_json:{channel_part}:{message_id}"

    def _edge_id(self, source_id: str, target_id: str, message_id: str) -> str:
        raw = "|".join([SourceProject.DISCORD_JSON.value, EdgeRelation.REFERENCES.value, source_id, target_id, message_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"discord-json-references-{digest}"

    def _fallback_message_id(self, path: Path, index: int, content: str) -> str:
        raw = f"{path.as_posix()}:{index}:{content}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _message_id(self, message: dict[str, Any]) -> str:
        return self._first(message, "id", "message_id", "messageId", "ID", "Message ID")

    def _message_index_key(self, value: Any) -> str:
        return self._string(value)

    def _content(self, message: dict[str, Any]) -> str:
        return self._first(message, "content", "Contents", "text", "message")

    def _title(self, channel: str, created_at: datetime, author: str) -> str:
        speaker = author or "unknown"
        channel_name = channel or "unknown-channel"
        return f"#{channel_name} {created_at.date().isoformat()} {speaker}"

    def _tag_value(self, value: str) -> str:
        return "-".join(value.lower().split())

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = self._string(value)
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.fromtimestamp(float(text), tz=timezone.utc)
            except ValueError:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _relative_path(self, path: Path, root: Path) -> str:
        source_root = root.parent if root.is_file() else root
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _object_value(self, value: Any, *keys: str) -> str:
        if not isinstance(value, dict):
            return ""
        return self._first(value, *keys)

    def _first(self, mapping: dict[str, Any], *keys: str) -> str:
        for key in keys:
            text = self._string(mapping.get(key))
            if text:
                return text
        return ""

    def _string(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()
