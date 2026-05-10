"""Adapter for Instagram archive exports."""

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
MENTION_RE = re.compile(r"(?<![\w/])@([A-Za-z0-9_.]+)")


class InstagramArchiveAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "instagram_archive"

    @property
    def entity_types(self) -> list[str]:
        return ["post", "story", "message"]

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
        ingest_stories = not entity_types or "story" in entity_types
        ingest_messages = not entity_types or "message" in entity_types

        # Track references for creating edges
        post_hashtags: dict[str, list[str]] = {}
        post_mentions: dict[str, list[str]] = {}
        post_locations: dict[str, dict[str, str]] = {}
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
                    # Track hashtags and mentions for edges
                    hashtags = self._extract_hashtags(post_data)
                    if hashtags:
                        post_hashtags[unit.source_id] = hashtags
                    mentions = self._extract_mentions(post_data)
                    if mentions:
                        post_mentions[unit.source_id] = mentions
                    location = self._extract_location(post_data)
                    if location:
                        post_locations[unit.source_id] = location

        if ingest_stories:
            for path in self._find_story_files(root):
                for story_data in self._read_stories(path):
                    unit = self._story_unit(story_data, path, root)
                    if unit is None:
                        continue
                    if sync_at and unit.updated_at <= sync_at:
                        continue
                    result.units.append(unit)

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

        # Create edges
        emitted_edges: set[tuple[str, str, str]] = set()

        # Hashtag edges
        for source_id in sorted(post_hashtags):
            if source_id not in included_source_ids:
                continue
            for hashtag in post_hashtags[source_id]:
                edge_key = (source_id, hashtag, "instagram_hashtag")
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(source_id, hashtag, "instagram_hashtag"),
                        from_unit_id=source_id,
                        to_unit_id=f"instagram_hashtag:{hashtag}",
                        relation=EdgeRelation.REFERENCES,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.INSTAGRAM_ARCHIVE.value,
                            "from_entity_type": "post",
                            "to_entity_type": "hashtag",
                            "relation_type": "instagram_hashtag",
                            "hashtag": hashtag,
                        },
                    )
                )

        # Mention edges
        for source_id in sorted(post_mentions):
            if source_id not in included_source_ids:
                continue
            for mention in post_mentions[source_id]:
                edge_key = (source_id, mention, "instagram_mention")
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(source_id, mention, "instagram_mention"),
                        from_unit_id=source_id,
                        to_unit_id=f"instagram_user:{mention}",
                        relation=EdgeRelation.REFERENCES,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.INSTAGRAM_ARCHIVE.value,
                            "from_entity_type": "post",
                            "to_entity_type": "user",
                            "relation_type": "instagram_mention",
                            "username": mention,
                        },
                    )
                )

        # Location edges
        for source_id in sorted(post_locations):
            if source_id not in included_source_ids:
                continue
            location = post_locations[source_id]
            location_name = location.get("name", "")
            if not location_name:
                continue
            edge_key = (source_id, location_name, "instagram_location")
            if edge_key in emitted_edges:
                continue
            emitted_edges.add(edge_key)
            result.edges.append(
                KnowledgeEdge(
                    id=self._edge_id(source_id, location_name, "instagram_location"),
                    from_unit_id=source_id,
                    to_unit_id=f"instagram_location:{self._location_id(location)}",
                    relation=EdgeRelation.REFERENCES,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.INSTAGRAM_ARCHIVE.value,
                        "from_entity_type": "post",
                        "to_entity_type": "location",
                        "relation_type": "instagram_location",
                        "location_name": location_name,
                        **{k: v for k, v in location.items() if k != "name" and v},
                    },
                )
            )

        # Conversation edges
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
                edge_key = (from_id, to_id, "instagram_conversation")
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(from_id, to_id, "instagram_conversation"),
                        from_unit_id=from_id,
                        to_unit_id=to_id,
                        relation=EdgeRelation.REFERENCES,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.INSTAGRAM_ARCHIVE.value,
                            "from_entity_type": "message",
                            "to_entity_type": "message",
                            "relation_type": "instagram_conversation",
                            "conversation_id": conv_id,
                        },
                    )
                )

        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id))
        return result

    def _find_post_files(self, root: Path) -> list[Path]:
        """Find post JSON files in content/ directory."""
        patterns = [
            root / "content" / "posts_*.json",
            root / "posts_*.json",
        ]
        paths: list[Path] = []
        for pattern in patterns:
            parent = pattern.parent
            if not parent.exists():
                continue
            for path in parent.glob(pattern.name):
                if path.is_file():
                    paths.append(path)
        return sorted(set(paths))

    def _find_story_files(self, root: Path) -> list[Path]:
        """Find story JSON files in content/ directory."""
        patterns = [
            root / "content" / "stories.json",
            root / "stories.json",
        ]
        paths: list[Path] = []
        for pattern in patterns:
            if pattern.exists() and pattern.is_file():
                paths.append(pattern)
        return sorted(set(paths))

    def _find_message_files(self, root: Path) -> list[Path]:
        """Find message JSON files in messages/inbox/ directory."""
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
        """Read and parse posts from JSON file."""
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        # Instagram post exports are typically a list of post objects
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        # Handle potential nested structures
        if isinstance(parsed, dict):
            for key in ("posts", "data", "items"):
                nested = parsed.get(key)
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
        return [parsed] if isinstance(parsed, dict) else []

    def _read_stories(self, path: Path) -> list[dict[str, Any]]:
        """Read and parse stories from JSON file."""
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        # Instagram story exports
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            # Could be wrapped in "stories" or "ig_stories" key
            for key in ("stories", "ig_stories", "data", "items"):
                nested = parsed.get(key)
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
        return [parsed] if isinstance(parsed, dict) else []

    def _read_messages(self, path: Path) -> list[dict[str, Any]]:
        """Read and parse messages from JSON file."""
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
        """Create KnowledgeUnit from Instagram post."""
        # Extract caption/title
        caption = self._first(post, "title", "caption")
        media = self._extract_media(post)

        # Need at least caption or media
        if not caption and not media:
            return None

        # Extract timestamp
        creation_timestamp = post.get("creation_timestamp")
        created_at = self._parse_timestamp(creation_timestamp)

        # Extract location
        location = self._extract_location(post)

        # Extract hashtags and mentions from caption
        hashtags = self._extract_hashtags(post)
        mentions = self._extract_mentions(post)

        # Generate post ID
        post_id = self._first(post, "id", "media_id")
        if not post_id:
            raw = f"{path.as_posix()}:{caption or ''}:{creation_timestamp}"
            post_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

        source_id = f"instagram_archive:post:{post_id}"
        source_path = self._relative_path(path, root)

        # Build metadata
        metadata: dict[str, Any] = {
            "post_id": post_id,
            "creation_timestamp": creation_timestamp,
            "media": media,
            "location": location,
            "hashtags": hashtags,
            "mentions": mentions,
            "source_path": source_path,
        }

        # Build tags
        tag_names = [f"hashtag-{self._tag_value(tag)}" for tag in hashtags]
        all_tags = ["instagram", "instagram-post"] + tag_names

        # Create title
        display_title = caption or "Instagram post"
        if len(display_title) > 80:
            display_title = f"{display_title[:77].rstrip()}..."

        return KnowledgeUnit(
            source_project=SourceProject.INSTAGRAM_ARCHIVE,
            source_id=source_id,
            source_entity_type="post",
            title=display_title,
            content=self._post_content(caption, location, media),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [], {})},
            tags=self._dedupe(all_tags),
            created_at=created_at,
            updated_at=created_at,
        )

    def _story_unit(
        self,
        story: dict[str, Any],
        path: Path,
        root: Path,
    ) -> KnowledgeUnit | None:
        """Create KnowledgeUnit from Instagram story."""
        # Extract media
        media = self._extract_media(story)
        if not media:
            return None

        # Extract timestamps
        creation_timestamp = story.get("creation_timestamp")
        created_at = self._parse_timestamp(creation_timestamp)
        expiration_timestamp = story.get("expiration_timestamp")

        # Generate story ID
        story_id = self._first(story, "id", "story_id")
        if not story_id:
            raw = f"{path.as_posix()}:{creation_timestamp}:{media}"
            story_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

        source_id = f"instagram_archive:story:{story_id}"
        source_path = self._relative_path(path, root)

        # Build metadata
        metadata: dict[str, Any] = {
            "story_id": story_id,
            "creation_timestamp": creation_timestamp,
            "expiration_timestamp": expiration_timestamp,
            "media": media,
            "source_path": source_path,
        }

        # Create title
        display_title = f"Instagram story {created_at.date().isoformat()}"

        return KnowledgeUnit(
            source_project=SourceProject.INSTAGRAM_ARCHIVE,
            source_id=source_id,
            source_entity_type="story",
            title=display_title,
            content=self._story_content(media, created_at, expiration_timestamp),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [], {})},
            tags=self._dedupe(["instagram", "instagram-story"]),
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
        """Create KnowledgeUnit from Instagram message."""
        content = self._first(message, "content", "text")
        sender_name = self._first(message, "sender_name")

        # Need at least content
        if not content:
            return None

        timestamp_ms = message.get("timestamp_ms")
        created_at = self._parse_timestamp_ms(timestamp_ms)

        # Extract media
        media_url = self._first(message, "media_url")
        media_type = self._first(message, "media_type")

        # Generate message ID
        message_id = self._first(message, "id", "message_id")
        if not message_id:
            raw = f"{conversation_id}:{sender_name}:{content}:{timestamp_ms}"
            message_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

        source_id = f"instagram_archive:message:{message_id}"
        source_path = self._relative_path(path, root)

        metadata: dict[str, Any] = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "sender_name": sender_name,
            "timestamp_ms": timestamp_ms,
            "media_url": media_url,
            "media_type": media_type,
            "source_path": source_path,
        }

        display_title = f"{sender_name}: {content}" if sender_name else content
        if len(display_title) > 80:
            display_title = f"{display_title[:77].rstrip()}..."

        return KnowledgeUnit(
            source_project=SourceProject.INSTAGRAM_ARCHIVE,
            source_id=source_id,
            source_entity_type="message",
            title=display_title,
            content=self._message_content(sender_name, content, media_url, media_type),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [], {})},
            tags=self._dedupe(["instagram", "instagram-message"]),
            created_at=created_at,
            updated_at=created_at,
        )

    def _extract_media(self, item: dict[str, Any]) -> list[dict[str, str]]:
        """Extract media information from post or story."""
        media: list[dict[str, str]] = []

        # Check for media array
        media_data = item.get("media")
        if isinstance(media_data, list):
            for media_item in media_data:
                if isinstance(media_item, dict):
                    media_info: dict[str, str] = {}
                    media_url = self._first(media_item, "uri", "url", "media_url")
                    if media_url:
                        media_info["media_url"] = media_url
                    media_type = self._first(media_item, "media_type", "type")
                    if media_type:
                        media_info["media_type"] = media_type
                    if media_info:
                        media.append(media_info)

        # Check for single media fields
        if not media:
            media_url = self._first(item, "uri", "url", "media_url")
            media_type = self._first(item, "media_type", "type")
            if media_url:
                media_info = {"media_url": media_url}
                if media_type:
                    media_info["media_type"] = media_type
                media.append(media_info)

        return media

    def _extract_location(self, post: dict[str, Any]) -> dict[str, str]:
        """Extract location information from post."""
        location_data = post.get("location")
        if not isinstance(location_data, dict):
            return {}

        location: dict[str, str] = {}
        name = self._first(location_data, "name")
        if name:
            location["name"] = name

        # Extract coordinates if available
        latitude = location_data.get("latitude")
        longitude = location_data.get("longitude")
        if latitude is not None and longitude is not None:
            location["latitude"] = str(latitude)
            location["longitude"] = str(longitude)

        return location

    def _extract_hashtags(self, post: dict[str, Any]) -> list[str]:
        """Extract hashtags from post caption."""
        caption = self._first(post, "title", "caption")
        if not caption:
            return []

        hashtags: set[str] = set()
        for match in HASHTAG_RE.finditer(caption):
            hashtags.add(match.group(1).lower())

        return sorted(hashtags)

    def _extract_mentions(self, post: dict[str, Any]) -> list[str]:
        """Extract username mentions from post caption."""
        caption = self._first(post, "title", "caption")
        if not caption:
            return []

        mentions: set[str] = set()
        for match in MENTION_RE.finditer(caption):
            mentions.add(match.group(1).lower())

        return sorted(mentions)

    def _conversation_id_from_path(self, path: Path) -> str:
        """Extract conversation ID from message file path."""
        if "inbox" in path.parts:
            inbox_index = path.parts.index("inbox")
            if inbox_index + 1 < len(path.parts):
                return path.parts[inbox_index + 1]
        return path.parent.name

    def _post_content(
        self,
        caption: str,
        location: dict[str, str],
        media: list[dict[str, str]],
    ) -> str:
        """Build content string for post."""
        parts = []
        if caption:
            parts.append(caption)
        if location and location.get("name"):
            parts.append(f"Location: {location['name']}")
        if media:
            media_types = [m.get("media_type", "media") for m in media]
            parts.append(f"Media: {len(media)} items ({', '.join(media_types)})")
        return "\n".join(parts) if parts else ""

    def _story_content(
        self,
        media: list[dict[str, str]],
        created_at: datetime,
        expiration_timestamp: Any,
    ) -> str:
        """Build content string for story."""
        parts = []
        parts.append(f"Created: {created_at.isoformat()}")
        if expiration_timestamp:
            expiration = self._parse_timestamp(expiration_timestamp)
            parts.append(f"Expires: {expiration.isoformat()}")
        if media:
            media_types = [m.get("media_type", "media") for m in media]
            parts.append(f"Media: {len(media)} items ({', '.join(media_types)})")
        return "\n".join(parts)

    def _message_content(
        self,
        sender: str,
        content: str,
        media_url: str,
        media_type: str,
    ) -> str:
        """Build content string for message."""
        parts = []
        if sender:
            parts.append(f"From: {sender}")
        parts.append(content)
        if media_url:
            media_desc = f"Media: {media_type}" if media_type else "Media attached"
            parts.append(media_desc)
        return "\n".join(parts)

    def _location_id(self, location: dict[str, str]) -> str:
        """Generate location ID from location data."""
        name = location.get("name", "")
        lat = location.get("latitude", "")
        lon = location.get("longitude", "")
        raw = f"{name}:{lat}:{lon}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _edge_id(self, source_id: str, target_ref: str, relation_type: str) -> str:
        """Generate edge ID."""
        raw = "|".join([SourceProject.INSTAGRAM_ARCHIVE.value, relation_type, source_id, target_ref])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"instagram-archive-{relation_type}-{digest}"

    def _relative_path(self, path: Path, root: Path) -> str:
        """Get relative path from root."""
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()

    def _tag_value(self, value: str) -> str:
        """Normalize tag value."""
        return "-".join(value.lower().split())

    def _dedupe(self, values: list[str]) -> list[str]:
        """Remove duplicates while preserving order."""
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                output.append(value)
        return output

    def _first(self, mapping: dict[str, Any], *keys: str) -> str:
        """Get first non-empty value from mapping."""
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
        """Convert SyncState to datetime."""
        value = since.last_sync_at
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
