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
        return ["slack_message", "reaction", "slack_thread"]

    def __init__(self, path: str = "", *, root_path: str = "") -> None:
        self.path = path or root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed = set(entity_types or self.entity_types)
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
        thread_ts = self._string(message.get("thread_ts"))
        links = self._extract_links(text)
        source_path = self._relative_path(path, root)

        metadata: dict[str, Any] = {
            "channel": channel,
            "user": user,
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

    def _thread_summary_edge_id(self, from_id: str, to_id: str) -> str:
        raw = "|".join(
            [SourceProject.SLACK_JSON.value, EdgeRelation.CONTAINS.value, from_id, to_id]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"slack-json-thread-contains-{digest[:16]}"

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
