"""Adapter for exported ChatGPT conversation JSON."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


@dataclass(frozen=True)
class _ChatMessage:
    node_id: str
    message_id: str
    role: str
    text: str
    created_at: datetime | None
    updated_at: datetime | None
    order: int


class ChatGptJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "chatgpt_json"

    @property
    def entity_types(self) -> list[str]:
        return ["chatgpt_conversation"]

    def __init__(self, path: str = "", *, root_path: str = "") -> None:
        self.path = path or root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "chatgpt_conversation" not in entity_types:
            return result

        sync_at = self._sync_datetime(since) if since else None
        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        for path in self._json_files(root):
            for conversation in self._read_conversations(path):
                messages = self._messages(conversation)
                unit = self._unit_from_conversation(conversation, path, root, messages)
                if unit is None:
                    continue
                comparable_at = unit.updated_at or unit.created_at
                if sync_at and comparable_at <= sync_at:
                    continue
                result.units.append(unit)
                result.edges.extend(self._message_edges(conversation, unit, messages))

        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: (edge.created_at, edge.id))
        return result

    def _json_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() == ".json" else []
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.json") if path.is_file())

    def _read_conversations(self, path: Path) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []

        return self._conversation_records(parsed)

    def _conversation_records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if self._is_conversation(item)]
        if not isinstance(value, dict):
            return []
        if self._is_conversation(value):
            return [value]
        for key in ("conversations", "items", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if self._is_conversation(item)]
        return []

    def _is_conversation(self, value: Any) -> bool:
        return isinstance(value, dict) and isinstance(value.get("mapping"), dict)

    def _unit_from_conversation(
        self,
        conversation: dict[str, Any],
        path: Path,
        root: Path,
        messages: list[_ChatMessage] | None = None,
    ) -> KnowledgeUnit | None:
        messages = messages if messages is not None else self._messages(conversation)
        if not messages:
            return None

        transcript = self._transcript(messages)
        if not transcript:
            return None

        conversation_id = self._conversation_id(conversation, path, transcript)
        created_at = (
            self._parse_datetime(conversation.get("create_time"))
            or self._first_datetime(message.created_at for message in messages)
            or datetime.now(timezone.utc)
        )
        updated_at = (
            self._parse_datetime(conversation.get("update_time"))
            or self._last_datetime(message.updated_at or message.created_at for message in messages)
            or created_at
        )
        roles = self._roles(messages)
        source_path = self._relative_path(path, root)

        return KnowledgeUnit(
            source_project=SourceProject.CHATGPT_JSON,
            source_id=f"chatgpt_json:{conversation_id}",
            source_entity_type="chatgpt_conversation",
            title=self._string(conversation.get("title")) or "Untitled ChatGPT conversation",
            content=transcript,
            content_type=ContentType.ARTIFACT,
            metadata={
                "conversation_id": conversation_id,
                "title": self._string(conversation.get("title")),
                "create_time": conversation.get("create_time"),
                "update_time": conversation.get("update_time"),
                "author_roles": roles,
                "message_count": len(messages),
                "message_ids": [message.message_id for message in messages if message.message_id],
                "source_file": path.name,
                "source_path": source_path,
                "path": source_path,
            },
            tags=["chatgpt"],
            created_at=created_at,
            updated_at=updated_at,
        )

    def _message_edges(
        self,
        conversation: dict[str, Any],
        unit: KnowledgeUnit,
        messages: list[_ChatMessage],
    ) -> list[KnowledgeEdge]:
        mapping = conversation.get("mapping")
        if not isinstance(mapping, dict):
            return []

        conversation_id = self._string(unit.metadata.get("conversation_id"))
        by_node = {message.node_id: message for message in messages}
        edges: list[KnowledgeEdge] = []
        emitted: set[tuple[str, str]] = set()
        for child in sorted(messages, key=lambda message: (message.order, message.node_id)):
            node = mapping.get(child.node_id)
            if not isinstance(node, dict):
                continue
            parent_node_id = self._string(node.get("parent"))
            if not parent_node_id or parent_node_id == child.node_id:
                continue
            parent = by_node.get(parent_node_id)
            if parent is None:
                continue

            from_id = self._message_source_id(conversation_id, child)
            to_id = self._message_source_id(conversation_id, parent)
            edge_key = (from_id, to_id)
            if edge_key in emitted:
                continue
            emitted.add(edge_key)
            edges.append(
                KnowledgeEdge(
                    id=self._edge_id(from_id, to_id),
                    from_unit_id=from_id,
                    to_unit_id=to_id,
                    relation=EdgeRelation.REPLIES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.CHATGPT_JSON.value,
                        "conversation_id": conversation_id,
                        "conversation_source_id": unit.source_id,
                        "parent_node_id": parent.node_id,
                        "child_node_id": child.node_id,
                        "parent_message_id": parent.message_id,
                        "child_message_id": child.message_id,
                        "parent_role": parent.role,
                        "child_role": child.role,
                    },
                    created_at=child.created_at or unit.created_at,
                )
            )
        return edges

    def _message_source_id(self, conversation_id: str, message: _ChatMessage) -> str:
        raw = f"{conversation_id}|{message.node_id}|{message.message_id}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"chatgpt_json:{conversation_id}:message:{digest[:16]}"

    def _edge_id(self, from_id: str, to_id: str) -> str:
        raw = "|".join([SourceProject.CHATGPT_JSON.value, EdgeRelation.REPLIES_TO.value, from_id, to_id])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"chatgpt-json-replies-{digest[:16]}"

    def _messages(self, conversation: dict[str, Any]) -> list[_ChatMessage]:
        mapping = conversation.get("mapping")
        if not isinstance(mapping, dict):
            return []

        order = self._node_order(mapping)
        messages: list[_ChatMessage] = []
        for index, node_id in enumerate(order):
            node = mapping.get(node_id)
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            text = self._message_text(message)
            if not text:
                continue
            role = self._role(message)
            messages.append(
                _ChatMessage(
                    node_id=node_id,
                    message_id=self._string(message.get("id")) or node_id,
                    role=role or "unknown",
                    text=text,
                    created_at=self._parse_datetime(message.get("create_time")),
                    updated_at=self._parse_datetime(message.get("update_time")),
                    order=index,
                )
            )

        return sorted(
            messages,
            key=lambda message: (
                message.created_at.timestamp() if message.created_at else float("inf"),
                message.order,
                message.node_id,
            ),
        )

    def _node_order(self, mapping: dict[str, Any]) -> list[str]:
        roots = sorted(
            node_id
            for node_id, node in mapping.items()
            if not isinstance(node, dict)
            or not node.get("parent")
            or str(node.get("parent")) not in mapping
        )
        if not roots:
            roots = sorted(mapping)

        ordered: list[str] = []
        seen: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in seen:
                return
            seen.add(node_id)
            ordered.append(node_id)
            node = mapping.get(node_id)
            if not isinstance(node, dict):
                return
            children = [str(child) for child in node.get("children", []) if str(child) in mapping]
            for child_id in sorted(children, key=self._node_sort_key(mapping)):
                visit(child_id)

        for root in roots:
            visit(str(root))
        for node_id in sorted(mapping):
            visit(str(node_id))
        return ordered

    def _node_sort_key(self, mapping: dict[str, Any]):
        def key(node_id: str) -> tuple[float, str]:
            node = mapping.get(node_id)
            message = node.get("message") if isinstance(node, dict) else None
            created_at = None
            if isinstance(message, dict):
                created_at = self._parse_datetime(message.get("create_time"))
            return (created_at.timestamp() if created_at else float("inf"), node_id)

        return key

    def _message_text(self, message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, dict):
            parts = content.get("parts")
            if isinstance(parts, list):
                return "\n".join(
                    text
                    for part in parts
                    if (text := self._part_text(part))
                ).strip()
            for key in ("text", "result", "summary"):
                text = self._part_text(content.get(key))
                if text:
                    return text
        return self._part_text(content)

    def _part_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("text", "content", "result"):
                text = self._part_text(value.get(key))
                if text:
                    return text
            return ""
        if isinstance(value, list):
            return "\n".join(text for item in value if (text := self._part_text(item))).strip()
        return str(value).strip()

    def _transcript(self, messages: list[_ChatMessage]) -> str:
        blocks = []
        for message in messages:
            label = message.role.replace("_", " ").title()
            blocks.append(f"{label}: {message.text}")
        return "\n\n".join(blocks).strip()

    def _conversation_id(
        self,
        conversation: dict[str, Any],
        path: Path,
        transcript: str,
    ) -> str:
        for key in ("id", "conversation_id"):
            value = self._string(conversation.get(key))
            if value:
                return value
        digest = hashlib.sha256(
            f"{path.name}\n{self._string(conversation.get('title'))}\n{transcript}".encode("utf-8")
        ).hexdigest()
        return digest[:24]

    def _roles(self, messages: list[_ChatMessage]) -> list[str]:
        roles: list[str] = []
        for message in messages:
            if message.role and message.role not in roles:
                roles.append(message.role)
        return roles

    def _role(self, message: dict[str, Any]) -> str:
        author = message.get("author")
        if not isinstance(author, dict):
            return ""
        return self._string(author.get("role"))

    def _first_datetime(self, values: Any) -> datetime | None:
        parsed = sorted(value for value in values if isinstance(value, datetime))
        return parsed[0] if parsed else None

    def _last_datetime(self, values: Any) -> datetime | None:
        parsed = sorted(value for value in values if isinstance(value, datetime))
        return parsed[-1] if parsed else None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, int | float):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        else:
            text = str(value).strip()
            try:
                number = float(text)
            except ValueError:
                number = None
            if number is not None:
                try:
                    return datetime.fromtimestamp(number, tz=timezone.utc)
                except (OSError, OverflowError, ValueError):
                    return None
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _relative_path(self, path: Path, root: Path) -> str:
        source_root = root.parent if root.is_file() else root
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _string(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()
