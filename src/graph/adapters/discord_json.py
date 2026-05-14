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
        return ["discord_message", "discord_attachment", "discord_channel"]

    def __init__(self, path: str = "", *, root_path: str = "") -> None:
        self.path = path or root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested_types = set(entity_types) if entity_types is not None else {"discord_message", "discord_attachment"}
        include_messages = "discord_message" in requested_types
        include_attachments = "discord_attachment" in requested_types
        include_channels = "discord_channel" in requested_types
        if not include_messages and not include_attachments and not include_channels:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        sync_at = self._sync_datetime(since) if since else None
        references_by_source_id: dict[str, list[dict[str, Any]]] = {}
        attachment_edges: list[KnowledgeEdge] = []
        message_units: list[KnowledgeUnit] = []

        for path in self._json_files(root):
            for index, record in enumerate(self._read_message_records(path)):
                unit = self._message_unit(record, path, root, index)
                if unit is None:
                    continue
                include_record = not sync_at or unit.updated_at > sync_at
                attachment_units = self._attachment_units(record, path, root, index, unit)
                if include_record and include_messages:
                    result.units.append(unit)
                    references_by_source_id[unit.source_id] = self._references(record["message"])
                if include_record and (include_messages or include_channels):
                    message_units.append(unit)
                if include_record and include_attachments:
                    result.units.extend(attachment_units)
                if include_record and include_messages and include_attachments:
                    attachment_edges.extend(
                        self._attachment_edge(unit, attachment_unit) for attachment_unit in attachment_units
                    )

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
                metadata: dict[str, Any] = {
                    "source_project": SourceProject.DISCORD_JSON.value,
                    "from_entity_type": "discord_message",
                    "to_entity_type": "discord_message",
                    "relation_type": "discord_reply_reference",
                    "referenced_message_id": referenced_message_id,
                    "referenced_channel_id": reference.get("channel_id", ""),
                    "referenced_server_id": reference.get("server_id", ""),
                }
                if reference.get("author"):
                    metadata["referenced_author"] = reference["author"]
                if reference.get("timestamp"):
                    metadata["referenced_timestamp"] = reference["timestamp"]
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(source_id, target_id, referenced_message_id),
                        from_unit_id=source_id,
                        to_unit_id=target_id,
                        relation=EdgeRelation.REFERENCES,
                        source=EdgeSource.SOURCE,
                        metadata=metadata,
                    )
                )

        for edge in attachment_edges:
            if edge.from_unit_id in included_source_ids and edge.to_unit_id in included_source_ids:
                result.edges.append(edge)

        if include_channels:
            channel_units = self._channel_units(message_units)
            result.units.extend(channel_units)
            if include_messages:
                result.edges.extend(self._channel_edges(channel_units, message_units))

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
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

    def _attachment_units(
        self,
        record: dict[str, Any],
        path: Path,
        root: Path,
        message_index: int,
        message_unit: KnowledgeUnit,
    ) -> list[KnowledgeUnit]:
        message = record["message"]
        attachments = self._attachments(message.get("attachments") or message.get("Attachments"))
        if not attachments:
            return []

        author = message_unit.metadata["author"]
        timestamp_text = message_unit.metadata["timestamp"]
        created_at = message_unit.created_at
        updated_at = message_unit.updated_at
        source_path = self._relative_path(path, root)
        channel_id = message_unit.metadata["channel_id"]
        channel_name = message_unit.metadata["channel_name"]
        server_id = message_unit.metadata["server_id"]
        server_name = message_unit.metadata["server_name"]
        message_id = message_unit.metadata["message_id"]

        units: list[KnowledgeUnit] = []
        for attachment_index, attachment in enumerate(attachments):
            if not self._has_attachment_identity(attachment):
                continue
            source_id = self._attachment_source_id(
                channel_id or channel_name,
                message_id,
                attachment,
                message_index,
                attachment_index,
            )
            title = attachment.get("filename") or attachment.get("url") or f"Discord attachment {attachment.get('id')}"
            metadata = {
                "attachment_id": attachment.get("id", ""),
                "filename": attachment.get("filename", ""),
                "url": attachment.get("url", ""),
                "content_type": attachment.get("content_type", ""),
                "size": attachment.get("size"),
                "message_id": message_id,
                "message_source_id": message_unit.source_id,
                "channel_id": channel_id,
                "channel_name": channel_name,
                "server_id": server_id,
                "server_name": server_name,
                "author": author,
                "timestamp": timestamp_text,
                "source_path": source_path,
                "path": source_path,
            }
            content_parts = [
                attachment.get("filename", ""),
                attachment.get("url", ""),
                attachment.get("content_type", ""),
            ]
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.DISCORD_JSON,
                    source_id=source_id,
                    source_entity_type="discord_attachment",
                    title=title,
                    content=" ".join(part for part in content_parts if part),
                    content_type=ContentType.ARTIFACT,
                    metadata={key: value for key, value in metadata.items() if value != "" and value is not None},
                    tags=self._attachment_tags(server_name, channel_name),
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _attachment_edge(self, message_unit: KnowledgeUnit, attachment_unit: KnowledgeUnit) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._edge_id(message_unit.source_id, attachment_unit.source_id, "discord_message_attachment"),
            from_unit_id=message_unit.source_id,
            to_unit_id=attachment_unit.source_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.DISCORD_JSON.value,
                "from_entity_type": "discord_message",
                "to_entity_type": "discord_attachment",
                "relation_type": "discord_message_attachment",
                "message_id": message_unit.metadata["message_id"],
                "attachment_id": attachment_unit.metadata.get("attachment_id", ""),
                "filename": attachment_unit.metadata.get("filename", ""),
                "url": attachment_unit.metadata.get("url", ""),
            },
        )

    def _channel_units(self, messages: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for message in messages:
            channel_key = self._channel_key(message)
            if channel_key:
                grouped.setdefault(channel_key, []).append(message)

        units: list[KnowledgeUnit] = []
        for channel_key, channel_messages in sorted(grouped.items()):
            ordered = sorted(channel_messages, key=lambda unit: (unit.created_at, self._string(unit.metadata.get("message_id")), unit.source_id))
            first = ordered[0]
            channel_id = self._string(first.metadata.get("channel_id"))
            channel_name = self._string(first.metadata.get("channel_name"))
            server_ids = sorted({self._string(unit.metadata.get("server_id")) for unit in ordered if self._string(unit.metadata.get("server_id"))})
            server_names = sorted({self._string(unit.metadata.get("server_name")) for unit in ordered if self._string(unit.metadata.get("server_name"))})
            authors = sorted({self._string((unit.metadata.get("author") or {}).get("id") or (unit.metadata.get("author") or {}).get("username")) for unit in ordered if isinstance(unit.metadata.get("author"), dict) and self._string((unit.metadata.get("author") or {}).get("id") or (unit.metadata.get("author") or {}).get("username"))})
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.DISCORD_JSON,
                    source_id=self._channel_source_id(channel_id or channel_name),
                    source_entity_type="discord_channel",
                    title=f"#{channel_name or channel_id or 'unknown-channel'}",
                    content=f"Discord channel #{channel_name or channel_id or 'unknown-channel'}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "channel_id": channel_id,
                        "channel_name": channel_name,
                        "message_count": len(ordered),
                        "attachment_count": sum(int(unit.metadata.get("attachment_count") or 0) for unit in ordered),
                        "author_count": len(authors),
                        "authors": authors,
                        "server_ids": server_ids,
                        "server_names": server_names,
                        "first_message_at": ordered[0].created_at.isoformat(),
                        "last_message_at": ordered[-1].created_at.isoformat(),
                        "source_paths": sorted({self._string(unit.metadata.get("source_path")) for unit in ordered if self._string(unit.metadata.get("source_path"))}),
                        "message_source_ids": [unit.source_id for unit in ordered],
                    },
                    tags=["discord", "discord-channel", *([f"discord-channel-{self._tag_value(channel_name)}"] if channel_name else [])],
                    created_at=ordered[0].created_at,
                    updated_at=max(unit.updated_at for unit in ordered),
                )
            )
        return units

    def _channel_edges(self, channels: list[KnowledgeUnit], messages: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        channel_ids = {
            self._channel_key(channel): channel.source_id
            for channel in channels
        }
        edges: list[KnowledgeEdge] = []
        for message in messages:
            channel_id = channel_ids.get(self._channel_key(message))
            if not channel_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=self._edge_id(channel_id, message.source_id, "discord_channel_message"),
                    from_unit_id=channel_id,
                    to_unit_id=message.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.DISCORD_JSON.value,
                        "from_entity_type": "discord_channel",
                        "to_entity_type": "discord_message",
                        "relation_type": "discord_channel_message",
                        "channel_id": self._string(message.metadata.get("channel_id")),
                        "channel_name": self._string(message.metadata.get("channel_name")),
                        "message_id": self._string(message.metadata.get("message_id")),
                    },
                )
            )
        return edges

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

    def _has_attachment_identity(self, attachment: dict[str, Any]) -> bool:
        return bool(attachment.get("id") or attachment.get("url") or attachment.get("filename"))

    def _references(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        values = [
            message.get("message_reference"),
            message.get("messageReference"),
            message.get("reference"),
            message.get("referenced_message"),
            message.get("referencedMessage"),
        ]
        references: list[dict[str, Any]] = []
        for value in values:
            reference = self._reference(value)
            if reference and reference not in references:
                references.append(reference)
        return references

    def _reference(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, str):
            message_id = value.strip()
            return {"message_id": message_id, "channel_id": "", "server_id": ""} if message_id else None
        if not isinstance(value, dict):
            return None
        message_id = self._first(value, "message_id", "messageId", "id")
        if not message_id:
            return None
        reference: dict[str, Any] = {
            "message_id": message_id,
            "channel_id": self._first(value, "channel_id", "channelId"),
            "server_id": self._first(value, "guild_id", "guildId", "server_id", "serverId"),
            "timestamp": self._first(value, "timestamp", "Timestamp", "created_at", "createdAt", "date"),
        }
        author = self._author(value.get("author") or value.get("Author"))
        if any(author.values()):
            reference["author"] = author
        return reference

    def _reference_source_id(
        self,
        reference: dict[str, Any],
        source_id_by_message_id: dict[str, str],
    ) -> str:
        return source_id_by_message_id.get(self._message_index_key(reference.get("message_id")), "")

    def _source_id(self, channel: str, message_id: str) -> str:
        channel_part = channel or "unknown-channel"
        return f"discord_json:{channel_part}:{message_id}"

    def _channel_source_id(self, channel: str) -> str:
        return f"discord_json:channel:{channel or 'unknown-channel'}"

    def _channel_key(self, unit: KnowledgeUnit) -> str:
        return self._string(unit.metadata.get("channel_id") or unit.metadata.get("channel_name"))

    def _attachment_source_id(
        self,
        channel: str,
        message_id: str,
        attachment: dict[str, Any],
        message_index: int,
        attachment_index: int,
    ) -> str:
        channel_part = channel or "unknown-channel"
        attachment_id = self._string(attachment.get("id"))
        if attachment_id:
            return f"discord_json:{channel_part}:{message_id}:attachment:{attachment_id}"
        raw = "|".join(
            [
                SourceProject.DISCORD_JSON.value,
                "discord_attachment",
                channel_part,
                message_id,
                self._string(attachment.get("url")),
                self._string(attachment.get("filename")),
                str(message_index),
                str(attachment_index),
            ]
        )
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"discord_json:{channel_part}:{message_id}:attachment:{digest}"

    def _attachment_tags(self, server_name: str, channel_name: str) -> list[str]:
        tags = ["discord", "discord-attachment"]
        if server_name:
            tags.append(f"discord-server-{self._tag_value(server_name)}")
        if channel_name:
            tags.append(f"discord-channel-{self._tag_value(channel_name)}")
        return tags

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
