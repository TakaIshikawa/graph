"""Adapter for Facebook archive exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


HASHTAG_RE = re.compile(r"(?<![\w/])#([\w][\w-]*)", re.UNICODE)
TAG_RE = re.compile(r"@\[(\d+):([^\]]+)\]")


class FacebookArchiveAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "facebook_archive"

    @property
    def entity_types(self) -> list[str]:
        return ["post", "message"]

    def __init__(self, path: str = "", *, root_path: str = "") -> None:
        self.path = path or root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        root = Path(self.path).expanduser()
        if not root.exists() or not root.is_dir():
            return result

        sync_at = self._sync_datetime(since) if since else None
        ingest_posts = not entity_types or "post" in entity_types
        ingest_messages = not entity_types or "message" in entity_types

        tagged_people: dict[str, list[dict[str, str]]] = {}
        conversations: dict[str, str] = {}

        if ingest_posts:
            for path in self._find_post_files(root):
                for post_data in self._read_posts(path):
                    unit = self._post_unit(post_data, path, root)
                    if unit is None:
                        continue
                    if sync_at and unit.updated_at <= sync_at:
                        continue
                    result.units.append(unit)
                    tags = self._extract_tags(post_data)
                    if tags:
                        tagged_people[unit.source_id] = tags

        if ingest_messages:
            for path in self._find_message_files(root):
                conversation_id = self._conversation_id_from_path(path)
                for message_data in self._read_messages(path):
                    unit = self._message_unit(message_data, path, root, conversation_id)
                    if unit is None:
                        continue
                    if sync_at and unit.updated_at <= sync_at:
                        continue
                    result.units.append(unit)
                    conversations[unit.source_id] = conversation_id

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        included_source_ids = {unit.source_id for unit in result.units}

        # Create edges for tagged people
        emitted_edges: set[tuple[str, str, str]] = set()
        for source_id in sorted(tagged_people):
            if source_id not in included_source_ids:
                continue
            for tag in tagged_people[source_id]:
                edge_key = (source_id, tag["name"], "facebook_tag")
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(source_id, tag["name"], "facebook_tag"),
                        from_unit_id=source_id,
                        to_unit_id=f"facebook_person:{tag['id']}",
                        relation=EdgeRelation.REFERENCES,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.FACEBOOK_ARCHIVE.value,
                            "from_entity_type": "post",
                            "to_entity_type": "person",
                            "relation_type": "facebook_tag",
                            "tagged_person_name": tag["name"],
                            "tagged_person_id": tag["id"],
                        },
                    )
                )

        # Create edges for conversations
        conversation_groups: dict[str, list[str]] = {}
        for source_id, conv_id in conversations.items():
            if source_id not in included_source_ids:
                continue
            conversation_groups.setdefault(conv_id, []).append(source_id)

        for conv_id, message_ids in conversation_groups.items():
            if len(message_ids) < 2:
                continue
            message_ids.sort()
            for i in range(len(message_ids) - 1):
                from_id = message_ids[i + 1]
                to_id = message_ids[i]
                edge_key = (from_id, to_id, "facebook_conversation")
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(from_id, to_id, "facebook_conversation"),
                        from_unit_id=from_id,
                        to_unit_id=to_id,
                        relation=EdgeRelation.REFERENCES,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.FACEBOOK_ARCHIVE.value,
                            "from_entity_type": "message",
                            "to_entity_type": "message",
                            "relation_type": "facebook_conversation",
                            "conversation_id": conv_id,
                        },
                    )
                )

        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id))
        return result

    def _find_post_files(self, root: Path) -> list[Path]:
        patterns = [
            root / "posts" / "your_posts_*.json",
            root / "posts" / "*.json",
            root / "your_posts_*.json",
        ]
        paths: list[Path] = []
        for pattern in patterns:
            if "*" in str(pattern):
                paths.extend(root.glob(str(pattern.relative_to(root))))
            elif pattern.exists():
                paths.append(pattern)
        return sorted(set(paths))

    def _find_message_files(self, root: Path) -> list[Path]:
        messages_root = root / "messages" / "inbox"
        if not messages_root.exists():
            return []
        paths: list[Path] = []
        for conversation_dir in messages_root.iterdir():
            if not conversation_dir.is_dir():
                continue
            for message_file in conversation_dir.glob("*.json"):
                paths.append(message_file)
        return sorted(paths)

    def _read_posts(self, path: Path) -> list[dict[str, Any]]:
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            # Handle nested structure
            for key in ("posts", "status_updates", "data"):
                nested = parsed.get(key)
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
        return [parsed] if isinstance(parsed, dict) else []

    def _read_messages(self, path: Path) -> list[dict[str, Any]]:
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict):
            messages = parsed.get("messages")
            if isinstance(messages, list):
                return [item for item in messages if isinstance(item, dict)]
        return []

    def _post_unit(
        self,
        post: dict[str, Any],
        path: Path,
        root: Path,
    ) -> KnowledgeUnit | None:
        content = self._decode_utf8(self._first(post, "post", "post_text", "data", "text"))
        title = self._first(post, "title")

        if not content and not title:
            return None

        timestamp = self._first(post, "timestamp")
        created_at = self._parse_timestamp(timestamp)

        attachments = self._extract_attachments(post)
        tags = self._extract_tags(post)
        place = self._extract_place(post)
        reactions = self._extract_reactions(post)
        comments_count = len(post.get("comments", []))

        post_id = self._first(post, "post_id", "id")
        if not post_id:
            raw = f"{path.as_posix()}:{content or title}:{timestamp}"
            post_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

        source_id = f"facebook_archive:post:{post_id}"
        source_path = self._relative_path(path, root)

        metadata: dict[str, Any] = {
            "post_id": post_id,
            "timestamp": timestamp,
            "attachments": attachments,
            "tags": [{"id": tag["id"], "name": tag["name"]} for tag in tags],
            "place": place,
            "reactions": reactions,
            "comments_count": comments_count,
            "source_path": source_path,
        }

        tag_names = [f"tag-{self._tag_value(tag['name'])}" for tag in tags]
        all_tags = ["facebook", "facebook-post"] + tag_names

        display_title = title or content or "Untitled Facebook post"
        if len(display_title) > 80:
            display_title = f"{display_title[:77].rstrip()}..."

        return KnowledgeUnit(
            source_project=SourceProject.FACEBOOK_ARCHIVE,
            source_id=source_id,
            source_entity_type="post",
            title=display_title,
            content=self._post_content(title, content, place, attachments),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [], {})},
            tags=self._dedupe(all_tags),
            created_at=created_at,
            updated_at=created_at,
        )

    def _message_unit(
        self,
        message: dict[str, Any],
        path: Path,
        root: Path,
        conversation_id: str,
    ) -> KnowledgeUnit | None:
        content = self._decode_utf8(self._first(message, "content", "text"))
        sender_name = self._decode_utf8(self._first(message, "sender_name"))

        if not content:
            return None

        timestamp_ms = message.get("timestamp_ms")
        created_at = self._parse_timestamp_ms(timestamp_ms)

        photos = self._string_list(message.get("photos"))
        videos = self._string_list(message.get("videos"))
        share = self._first(message, "share")
        reactions = message.get("reactions", [])

        message_id = self._first(message, "message_id", "id")
        if not message_id:
            raw = f"{conversation_id}:{sender_name}:{content}:{timestamp_ms}"
            message_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

        source_id = f"facebook_archive:message:{message_id}"
        source_path = self._relative_path(path, root)

        metadata: dict[str, Any] = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "sender_name": sender_name,
            "timestamp_ms": timestamp_ms,
            "photos": photos,
            "videos": videos,
            "share": share,
            "reactions": [self._decode_reaction(r) for r in reactions] if reactions else [],
            "source_path": source_path,
        }

        display_title = f"{sender_name}: {content}" if sender_name else content
        if len(display_title) > 80:
            display_title = f"{display_title[:77].rstrip()}..."

        return KnowledgeUnit(
            source_project=SourceProject.FACEBOOK_ARCHIVE,
            source_id=source_id,
            source_entity_type="message",
            title=display_title,
            content=self._message_content(sender_name, content, photos, videos, share),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [], {})},
            tags=self._dedupe(["facebook", "facebook-message"]),
            created_at=created_at,
            updated_at=created_at,
        )

    def _extract_attachments(self, post: dict[str, Any]) -> list[dict[str, str]]:
        attachments = []
        attachment_data = post.get("attachments", [])
        if not isinstance(attachment_data, list):
            return []

        for attachment in attachment_data:
            if not isinstance(attachment, dict):
                continue
            data = attachment.get("data")
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        attachments.append(self._process_attachment_data(item))
            elif isinstance(data, dict):
                attachments.append(self._process_attachment_data(data))

        return [a for a in attachments if a]

    def _process_attachment_data(self, data: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        media_uri = self._first(data, "media", "uri")
        if media_uri:
            result["media"] = media_uri
        # Handle external_context which is nested
        external_context = data.get("external_context")
        if isinstance(external_context, dict):
            external_uri = self._first(external_context, "url")
            if external_uri:
                result["url"] = external_uri
        name = self._decode_utf8(self._first(data, "name", "title"))
        if name:
            result["name"] = name
        return result

    def _extract_tags(self, post: dict[str, Any]) -> list[dict[str, str]]:
        tags = []
        tags_data = post.get("tags", [])
        if isinstance(tags_data, list):
            for tag in tags_data:
                if isinstance(tag, dict):
                    tag_id = self._first(tag, "id")
                    tag_name = self._decode_utf8(self._first(tag, "name"))
                    if tag_id and tag_name:
                        tags.append({"id": tag_id, "name": tag_name})

        # Also extract inline tags from content
        content = self._decode_utf8(self._first(post, "post", "post_text", "data", "text"))
        for match in TAG_RE.finditer(content):
            tag_id = match.group(1)
            tag_name = match.group(2)
            if not any(t["id"] == tag_id for t in tags):
                tags.append({"id": tag_id, "name": tag_name})

        return tags

    def _extract_place(self, post: dict[str, Any]) -> dict[str, str]:
        place_data = post.get("place")
        if not isinstance(place_data, dict):
            return {}
        return {
            "name": self._decode_utf8(self._first(place_data, "name")),
            "address": self._decode_utf8(self._first(place_data, "address")),
            "coordinate": self._first(place_data, "coordinate", "coordinates"),
        }

    def _extract_reactions(self, post: dict[str, Any]) -> dict[str, int]:
        reactions_data = post.get("reactions", [])
        if not isinstance(reactions_data, list):
            return {}

        reaction_counts: dict[str, int] = {}
        for reaction in reactions_data:
            if isinstance(reaction, dict):
                reaction_type = self._first(reaction, "reaction", "type")
                if reaction_type:
                    reaction_counts[reaction_type] = reaction_counts.get(reaction_type, 0) + 1

        return reaction_counts

    def _conversation_id_from_path(self, path: Path) -> str:
        # Extract conversation ID from path like messages/inbox/conversationname_xyz/message_1.json
        if "inbox" in path.parts:
            inbox_index = path.parts.index("inbox")
            if inbox_index + 1 < len(path.parts):
                return path.parts[inbox_index + 1]
        return path.parent.name

    def _post_content(
        self,
        title: str,
        content: str,
        place: dict[str, str],
        attachments: list[dict[str, str]],
    ) -> str:
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if content:
            parts.append(content)
        if place and place.get("name"):
            parts.append(f"Location: {place['name']}")
        if attachments:
            parts.append(f"Attachments: {len(attachments)} items")
        return "\n".join(parts)

    def _message_content(
        self,
        sender: str,
        content: str,
        photos: list[str],
        videos: list[str],
        share: str,
    ) -> str:
        parts = []
        if sender:
            parts.append(f"From: {sender}")
        parts.append(content)
        if photos:
            parts.append(f"Photos: {len(photos)}")
        if videos:
            parts.append(f"Videos: {len(videos)}")
        if share:
            parts.append(f"Shared: {share}")
        return "\n".join(parts)

    def _decode_utf8(self, value: str) -> str:
        """Decode Facebook's UTF-8 encoding where non-ASCII chars are encoded as \\uXXXX."""
        if not value:
            return ""
        try:
            # Facebook archives often have improperly encoded UTF-8
            # Try to decode bytes if needed
            return value.encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return value

    def _decode_reaction(self, reaction: dict[str, Any]) -> dict[str, str]:
        return {
            "actor": self._decode_utf8(self._first(reaction, "actor")),
            "reaction": self._first(reaction, "reaction"),
        }

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [
                self._first(item, "uri", "url") if isinstance(item, dict) else str(item)
                for item in value
                if item
            ]
        return []

    def _edge_id(self, source_id: str, target_ref: str, relation_type: str) -> str:
        raw = "|".join([SourceProject.FACEBOOK_ARCHIVE.value, relation_type, source_id, target_ref])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"facebook-archive-{relation_type}-{digest}"

    def _relative_path(self, path: Path, root: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()

    def _tag_value(self, value: str) -> str:
        return "-".join(value.lower().split())

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                output.append(value)
        return output

    def _first(self, mapping: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = mapping.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _parse_timestamp(self, value: Any) -> datetime:
        """Parse Unix timestamp (seconds)."""
        if value is None:
            return datetime.now(timezone.utc)
        try:
            timestamp = int(value)
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            return datetime.now(timezone.utc)

    def _parse_timestamp_ms(self, value: Any) -> datetime:
        """Parse Unix timestamp (milliseconds)."""
        if value is None:
            return datetime.now(timezone.utc)
        try:
            timestamp_ms = int(value)
            return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            return datetime.now(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
