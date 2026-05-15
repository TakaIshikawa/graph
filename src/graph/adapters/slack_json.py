"""Adapter for Slack JSON workspace exports."""

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


SLACK_LINK_RE = re.compile(r"<(?P<url>https?://[^>|]+)(?:\|(?P<label>[^>]+))?>")
PLAIN_URL_RE = re.compile(r"https?://[^\s<>()]+")


class SlackJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "slack_json"

    @property
    def entity_types(self) -> list[str]:
        return ["slack_message", "reaction", "slack_thread", "slack_channel", "user"]

    def __init__(self, path: str = "", *, root_path: str = "") -> None:
        self.path = path or root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types) if entity_types is not None else {"slack_message", "reaction", "slack_thread"}
        if not allowed.intersection(self.entity_types):
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        sync_at = self._sync_datetime(since) if since else None
        message_units: list[KnowledgeUnit] = []
        for path in self._json_files(root):
            channel = self._channel_name(root, path)
            for message in self._read_messages(path):
                unit = self._message_unit(message, channel, path, root)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                message_emitted = "slack_message" in allowed
                if message_emitted:
                    result.units.append(unit)
                if message_emitted or "slack_channel" in allowed or "slack_thread" in allowed or "user" in allowed:
                    message_units.append(unit)
                reaction_units = self._reaction_units(message, unit)
                if "reaction" in allowed:
                    result.units.extend(reaction_units)
                if message_emitted and "reaction" in allowed:
                    result.edges.extend(self._reaction_edges(reaction_units, unit))

        if "slack_message" in allowed:
            result.edges.extend(self._thread_edges(message_units))
        if {"slack_message", "slack_thread"}.issubset(allowed):
            summary_units, summary_edges = self._thread_summary_units(message_units)
            result.units.extend(summary_units)
            result.edges.extend(summary_edges)
        if "slack_channel" in allowed:
            channel_units = self._channel_units(message_units)
            result.units.extend(channel_units)
            if "slack_message" in allowed:
                result.edges.extend(self._channel_edges(channel_units, message_units))
        if "user" in allowed:
            user_units = self._user_units(message_units)
            result.units.extend(user_units)
            if "slack_message" in allowed:
                result.edges.extend(self._user_edges(user_units, message_units))
        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id))
        return result

    def _json_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() == ".json" else []
        if not root.is_dir():
            return []
        return sorted(path for path in root.glob("*.json") if path.is_file())

    def _read_messages(self, path: Path) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []

        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            if isinstance(parsed.get("messages"), list):
                return [item for item in parsed["messages"] if isinstance(item, dict)]
            if parsed.get("type") == "message" or "ts" in parsed:
                return [parsed]
        return []

    def _message_unit(
        self,
        message: dict[str, Any],
        channel: str,
        path: Path,
        root: Path,
    ) -> KnowledgeUnit | None:
        if message.get("subtype") in {"message_deleted", "tombstone"}:
            return None

        text = self._string(message.get("text"))
        content = self._readable_text(text)
        if not content:
            return None

        ts = self._string(message.get("ts"))
        if not ts:
            return None

        created_at = self._parse_slack_timestamp(ts)
        if created_at is None:
            return None
        updated_at = self._parse_slack_timestamp(self._string(message.get("edited", {}).get("ts")))
        if updated_at is None:
            updated_at = created_at

        user = self._user(message)
        user_metadata = self._user_metadata(message, user)
        thread_ts = self._string(message.get("thread_ts"))
        links = self._extract_links(text)
        source_path = self._relative_path(path, root)

        metadata: dict[str, Any] = {
            "channel": channel,
            "user": user,
            "user_id": user_metadata.get("user_id"),
            "user_display_name": user_metadata.get("display_name"),
            "user_profile": user_metadata,
            "ts": ts,
            "datetime": created_at.isoformat(),
            "source_path": source_path,
            "path": source_path,
            "links": links,
        }
        subtype = self._string(message.get("subtype"))
        if subtype:
            metadata["subtype"] = subtype
        if thread_ts:
            metadata["thread_ts"] = thread_ts
            metadata["is_thread_reply"] = thread_ts != ts
        if message.get("reply_count") is not None:
            metadata["reply_count"] = message.get("reply_count")
        if isinstance(message.get("reactions"), list):
            metadata["reactions"] = message.get("reactions")
            metadata["reaction_count"] = sum(
                reaction.get("count", 0)
                for reaction in message["reactions"]
                if isinstance(reaction, dict) and isinstance(reaction.get("count"), int)
            )
        if message.get("client_msg_id"):
            metadata["client_msg_id"] = self._string(message.get("client_msg_id"))

        return KnowledgeUnit(
            source_project=SourceProject.SLACK_JSON,
            source_id=self._source_id(channel, ts),
            source_entity_type="slack_message",
            title=self._title(channel, created_at, user),
            content=content,
            content_type=ContentType.INSIGHT,
            metadata=metadata,
            tags=["slack", f"slack-{channel}"],
            created_at=created_at,
            updated_at=updated_at,
        )

    def _reaction_units(
        self, message: dict[str, Any], parent: KnowledgeUnit
    ) -> list[KnowledgeUnit]:
        reactions = message.get("reactions")
        if not isinstance(reactions, list):
            return []

        channel = self._string(parent.metadata.get("channel"))
        channel_id = self._string(message.get("channel") or message.get("channel_id"))
        ts = self._string(parent.metadata.get("ts"))
        units: list[KnowledgeUnit] = []
        for reaction in sorted(
            (item for item in reactions if isinstance(item, dict)),
            key=lambda item: (self._string(item.get("name")), self._string(item.get("emoji"))),
        ):
            name = self._string(reaction.get("name") or reaction.get("emoji"))
            if not name:
                continue
            count = reaction.get("count")
            if not isinstance(count, int) or isinstance(count, bool):
                users = reaction.get("users")
                count = len(users) if isinstance(users, list) else 0
            users = (
                [self._string(user) for user in reaction.get("users", []) if self._string(user)]
                if isinstance(reaction.get("users"), list)
                else []
            )
            metadata = {
                "channel_id": channel_id,
                "channel_name": channel,
                "channel": channel,
                "message_ts": ts,
                "reaction_name": name,
                "reaction_count": count,
                "reacting_users": users,
                "parent_source_id": parent.source_id,
                "source_file": self._string(parent.metadata.get("source_path")),
            }
            thread_ts = self._string(parent.metadata.get("thread_ts"))
            if thread_ts:
                metadata["thread_ts"] = thread_ts
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.SLACK_JSON,
                    source_id=self._reaction_source_id(channel, ts, name),
                    source_entity_type="reaction",
                    title=f"#{channel} reaction :{name}:",
                    content=f"Reaction :{name}: on Slack message {ts} ({count})",
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=["slack", "reaction", f"slack-{channel}"],
                    created_at=parent.created_at,
                    updated_at=parent.updated_at,
                )
            )
        return units

    def _reaction_edges(self, reactions: list[KnowledgeUnit], parent: KnowledgeUnit) -> list[KnowledgeEdge]:
        return [
            KnowledgeEdge(
                id=self._reaction_edge_id(reaction.source_id, parent.source_id),
                from_unit_id=reaction.source_id,
                to_unit_id=parent.source_id,
                relation=EdgeRelation.REFERENCES,
                source=EdgeSource.SOURCE,
                metadata={
                    "source_project": SourceProject.SLACK_JSON.value,
                    "from_entity_type": "reaction",
                    "to_entity_type": "slack_message",
                    "channel": parent.metadata.get("channel"),
                    "message_ts": parent.metadata.get("ts"),
                    "reaction_name": reaction.metadata.get("reaction_name"),
                },
                created_at=parent.created_at,
            )
            for reaction in reactions
        ]

    def _channel_name(self, root: Path, path: Path) -> str:
        if root.is_file():
            return path.stem
        return root.name

    def _source_id(self, channel: str, ts: str) -> str:
        return f"slack_json:{channel}:{ts}"

    def _reaction_source_id(self, channel: str, ts: str, name: str) -> str:
        return f"slack_json:{channel}:{ts}:reaction:{name}"

    def _title(self, channel: str, created_at: datetime, user: str) -> str:
        speaker = user or "unknown"
        return f"#{channel} {created_at.date().isoformat()} {speaker}"

    def _user(self, message: dict[str, Any]) -> str:
        for key in ("user", "username", "bot_id"):
            value = self._string(message.get(key))
            if value:
                return value
        profile = message.get("user_profile")
        if isinstance(profile, dict):
            for key in ("real_name", "name"):
                value = self._string(profile.get(key))
                if value:
                    return value
        return ""

    def _user_metadata(self, message: dict[str, Any], user: str) -> dict[str, Any]:
        profile = message.get("user_profile")
        profile_data = profile if isinstance(profile, dict) else {}
        display_name = (
            self._string(profile_data.get("display_name"))
            or self._string(profile_data.get("real_name"))
            or self._string(profile_data.get("name"))
            or self._string(message.get("username"))
            or user
        )
        metadata = {
            "user_id": self._string(message.get("user") or message.get("bot_id") or user),
            "username": self._string(message.get("username") or profile_data.get("name")),
            "display_name": display_name,
            "real_name": self._string(profile_data.get("real_name")),
            "bot_id": self._string(message.get("bot_id")),
            "is_bot": bool(message.get("bot_id") or message.get("subtype") == "bot_message"),
        }
        return {key: value for key, value in metadata.items() if value not in ("", None)}

    def _readable_text(self, text: str) -> str:
        text = SLACK_LINK_RE.sub(
            lambda match: match.group("label") or match.group("url"),
            text,
        )
        return " ".join(text.split())

    def _extract_links(self, text: str) -> list[str]:
        links: list[str] = []
        for match in SLACK_LINK_RE.finditer(text):
            self._append_link(links, match.group("url"))
        plain_text = SLACK_LINK_RE.sub("", text)
        for match in PLAIN_URL_RE.finditer(plain_text):
            self._append_link(links, match.group(0).rstrip(".,;:!?]}'\""))
        return links

    def _append_link(self, links: list[str], url: str) -> None:
        if url and url not in links:
            links.append(url)

    def _thread_edges(self, units: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        source_by_message: dict[tuple[str, str], KnowledgeUnit] = {}
        for unit in units:
            channel = self._string(unit.metadata.get("channel"))
            ts = self._string(unit.metadata.get("ts"))
            if channel and ts:
                source_by_message[(channel, ts)] = unit

        edges: list[KnowledgeEdge] = []
        emitted: set[tuple[str, str]] = set()
        for unit in units:
            channel = self._string(unit.metadata.get("channel"))
            ts = self._string(unit.metadata.get("ts"))
            thread_ts = self._string(unit.metadata.get("thread_ts"))
            if not channel or not ts or not thread_ts or thread_ts == ts:
                continue
            root = source_by_message.get((channel, thread_ts))
            if root is None or root.source_id == unit.source_id:
                continue
            edge_key = (unit.source_id, root.source_id)
            if edge_key in emitted:
                continue
            emitted.add(edge_key)
            edges.append(
                KnowledgeEdge(
                    id=self._edge_id(unit.source_id, root.source_id),
                    from_unit_id=unit.source_id,
                    to_unit_id=root.source_id,
                    relation=EdgeRelation.REPLIES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.SLACK_JSON.value,
                        "from_entity_type": "slack_message",
                        "to_entity_type": "slack_message",
                        "relation_type": "thread_reply",
                        "channel": channel,
                        "ts": ts,
                        "thread_ts": thread_ts,
                        "root_ts": thread_ts,
                        "root_message_source_id": root.source_id,
                    },
                    created_at=unit.created_at,
                )
            )
        return edges

    def _thread_summary_units(
        self, units: list[KnowledgeUnit]
    ) -> tuple[list[KnowledgeUnit], list[KnowledgeEdge]]:
        threads: dict[tuple[str, str], list[KnowledgeUnit]] = {}
        for unit in units:
            channel = self._string(unit.metadata.get("channel"))
            ts = self._string(unit.metadata.get("ts"))
            thread_ts = self._string(unit.metadata.get("thread_ts"))
            if not channel or not ts or not thread_ts:
                continue
            threads.setdefault((channel, thread_ts), []).append(unit)

        summaries: list[KnowledgeUnit] = []
        edges: list[KnowledgeEdge] = []
        for (channel, thread_ts), thread_units in sorted(threads.items()):
            ordered = sorted(
                thread_units,
                key=lambda unit: (
                    unit.created_at,
                    self._string(unit.metadata.get("ts")),
                    unit.source_id,
                ),
            )
            if len(ordered) < 2:
                continue
            root = next(
                (unit for unit in ordered if self._string(unit.metadata.get("ts")) == thread_ts),
                ordered[0],
            )
            replies = [
                unit for unit in ordered if self._string(unit.metadata.get("ts")) != thread_ts
            ]
            if not replies:
                continue
            participants = sorted(
                {
                    self._string(unit.metadata.get("user"))
                    for unit in ordered
                    if self._string(unit.metadata.get("user"))
                }
            )
            started_at = min(unit.created_at for unit in ordered)
            ended_at = max(unit.created_at for unit in ordered)
            source_id = self._thread_summary_source_id(channel, thread_ts)
            message_unit_ids = [unit.source_id for unit in ordered]
            summary = KnowledgeUnit(
                source_project=SourceProject.SLACK_JSON,
                source_id=source_id,
                source_entity_type="slack_thread",
                title=f"#{channel} thread {thread_ts}",
                content=self._thread_summary_content(ordered),
                content_type=ContentType.INSIGHT,
                metadata={
                    "channel": channel,
                    "thread_ts": thread_ts,
                    "participant_count": len(participants),
                    "participants": participants,
                    "reply_count": len(replies),
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "message_unit_ids": message_unit_ids,
                    "root_message_unit_id": root.source_id,
                },
                tags=["slack", "slack-thread", f"slack-{channel}"],
                created_at=started_at,
                updated_at=ended_at,
            )
            summaries.append(summary)
            for unit in ordered:
                edges.append(
                    KnowledgeEdge(
                        id=self._thread_summary_edge_id(source_id, unit.source_id),
                        from_unit_id=source_id,
                        to_unit_id=unit.source_id,
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.SLACK_JSON.value,
                            "from_entity_type": "slack_thread",
                            "to_entity_type": "slack_message",
                            "channel": channel,
                            "thread_ts": thread_ts,
                            "message_ts": self._string(unit.metadata.get("ts")),
                        },
                        created_at=unit.created_at,
                    )
                )
        return summaries, edges

    def _channel_units(self, units: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for unit in units:
            channel = self._string(unit.metadata.get("channel"))
            if channel:
                grouped.setdefault(channel, []).append(unit)

        channels: list[KnowledgeUnit] = []
        for channel, channel_units in sorted(grouped.items()):
            ordered = sorted(channel_units, key=lambda unit: (unit.created_at, self._string(unit.metadata.get("ts")), unit.source_id))
            users = sorted({self._string(unit.metadata.get("user")) for unit in ordered if self._string(unit.metadata.get("user"))})
            source_paths = sorted({self._string(unit.metadata.get("source_path")) for unit in ordered if self._string(unit.metadata.get("source_path"))})
            channels.append(
                KnowledgeUnit(
                    source_project=SourceProject.SLACK_JSON,
                    source_id=self._channel_source_id(channel),
                    source_entity_type="slack_channel",
                    title=f"#{channel}",
                    content=f"Slack channel #{channel}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "channel": channel,
                        "message_count": len(ordered),
                        "participant_count": len(users),
                        "first_message_at": ordered[0].created_at.isoformat(),
                        "last_message_at": ordered[-1].created_at.isoformat(),
                        "source_paths": source_paths,
                        "users": users,
                        "message_source_ids": [unit.source_id for unit in ordered],
                    },
                    tags=["slack", "slack-channel", f"slack-{channel}"],
                    created_at=ordered[0].created_at,
                    updated_at=max(unit.updated_at for unit in ordered),
                )
            )
        return channels

    def _channel_edges(self, channels: list[KnowledgeUnit], messages: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        channel_ids = {self._string(channel.metadata["channel"]): channel.source_id for channel in channels}
        edges: list[KnowledgeEdge] = []
        for message in messages:
            channel = self._string(message.metadata.get("channel"))
            channel_id = channel_ids.get(channel)
            if not channel_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=self._channel_edge_id(channel_id, message.source_id),
                    from_unit_id=channel_id,
                    to_unit_id=message.source_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.SLACK_JSON.value,
                        "from_entity_type": "slack_channel",
                        "to_entity_type": "slack_message",
                        "channel": channel,
                        "message_ts": self._string(message.metadata.get("ts")),
                    },
                    created_at=message.created_at,
                )
            )
        return edges

    def _user_units(self, units: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for unit in units:
            key = self._user_key(unit)
            if key:
                grouped.setdefault(key, []).append(unit)

        users: list[KnowledgeUnit] = []
        for key, user_messages in sorted(grouped.items()):
            ordered = sorted(user_messages, key=lambda unit: (unit.created_at, self._string(unit.metadata.get("ts")), unit.source_id))
            profile = ordered[0].metadata.get("user_profile") if isinstance(ordered[0].metadata.get("user_profile"), dict) else {}
            user_profile = profile if isinstance(profile, dict) else {}
            display_name = self._string(user_profile.get("display_name") or ordered[0].metadata.get("user") or key)
            channels = sorted({self._string(unit.metadata.get("channel")) for unit in ordered if self._string(unit.metadata.get("channel"))})
            users.append(
                KnowledgeUnit(
                    source_project=SourceProject.SLACK_JSON,
                    source_id=self._user_source_id(key),
                    source_entity_type="user",
                    title=display_name,
                    content=f"Slack user: {display_name}\nMessages: {len(ordered)}",
                    content_type=ContentType.METADATA,
                    metadata={
                        "user": self._string(user_profile.get("user_id") or ordered[0].metadata.get("user")),
                        "user_key": key,
                        "user_profile": user_profile,
                        "message_count": len(ordered),
                        "message_source_ids": [unit.source_id for unit in ordered],
                        "channels": channels,
                        "first_message_at": ordered[0].created_at.isoformat(),
                        "last_message_at": ordered[-1].created_at.isoformat(),
                    },
                    tags=["slack", "user"],
                    created_at=ordered[0].created_at,
                    updated_at=max(unit.updated_at for unit in ordered),
                )
            )
        return users

    def _user_edges(self, users: list[KnowledgeUnit], messages: list[KnowledgeUnit]) -> list[KnowledgeEdge]:
        user_ids = {self._string(user.metadata.get("user_key")): user.source_id for user in users}
        edges: list[KnowledgeEdge] = []
        for message in messages:
            key = self._user_key(message)
            user_id = user_ids.get(key)
            if not user_id:
                continue
            edges.append(
                KnowledgeEdge(
                    id=self._user_edge_id(message.source_id, user_id),
                    from_unit_id=message.source_id,
                    to_unit_id=user_id,
                    relation=EdgeRelation.RELATES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.SLACK_JSON.value,
                        "from_entity_type": "slack_message",
                        "to_entity_type": "user",
                        "relation_type": "message_user",
                        "channel": self._string(message.metadata.get("channel")),
                        "message_ts": self._string(message.metadata.get("ts")),
                        "user": self._string(message.metadata.get("user")),
                    },
                    created_at=message.created_at,
                )
            )
        return edges

    def _edge_id(self, from_id: str, to_id: str) -> str:
        raw = "|".join(
            [SourceProject.SLACK_JSON.value, EdgeRelation.REPLIES_TO.value, from_id, to_id]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"slack-json-replies-{digest[:16]}"

    def _reaction_edge_id(self, from_id: str, to_id: str) -> str:
        raw = "|".join(
            [SourceProject.SLACK_JSON.value, EdgeRelation.REFERENCES.value, from_id, to_id]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"slack-json-reaction-references-{digest[:16]}"

    def _thread_summary_source_id(self, channel: str, thread_ts: str) -> str:
        return f"slack_json:{channel}:{thread_ts}:thread_summary"

    def _channel_source_id(self, channel: str) -> str:
        return f"slack_json:channel:{channel}"

    def _thread_summary_edge_id(self, from_id: str, to_id: str) -> str:
        raw = "|".join(
            [SourceProject.SLACK_JSON.value, EdgeRelation.CONTAINS.value, from_id, to_id]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"slack-json-thread-contains-{digest[:16]}"

    def _channel_edge_id(self, from_id: str, to_id: str) -> str:
        raw = "|".join(
            [SourceProject.SLACK_JSON.value, "slack_channel", EdgeRelation.CONTAINS.value, from_id, to_id]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"slack-json-channel-contains-{digest[:16]}"

    def _user_key(self, unit: KnowledgeUnit) -> str:
        profile = unit.metadata.get("user_profile") if isinstance(unit.metadata.get("user_profile"), dict) else {}
        user_id = self._string(unit.metadata.get("user_id") or (profile or {}).get("user_id") or unit.metadata.get("user"))
        return user_id.casefold()

    def _user_source_id(self, user_key: str) -> str:
        digest = hashlib.sha256(user_key.encode("utf-8")).hexdigest()[:24]
        return f"slack_json:user:{digest}"

    def _user_edge_id(self, from_id: str, to_id: str) -> str:
        raw = "|".join([SourceProject.SLACK_JSON.value, "user", EdgeRelation.RELATES_TO.value, from_id, to_id])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"slack-json-user-relates-{digest[:16]}"

    def _thread_summary_content(self, units: list[KnowledgeUnit]) -> str:
        lines: list[str] = []
        for unit in units[:5]:
            user = self._string(unit.metadata.get("user")) or "unknown"
            text = " ".join(unit.content.split())
            if len(text) > 160:
                text = f"{text[:157].rstrip()}..."
            lines.append(f"{user}: {text}")
        return "\n".join(lines)

    def _parse_slack_timestamp(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except ValueError:
            return None

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
