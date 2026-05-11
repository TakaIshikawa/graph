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
        return ["slack_message"]

    def __init__(self, path: str = "", *, root_path: str = "") -> None:
        self.path = path or root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "slack_message" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        sync_at = self._sync_datetime(since) if since else None
        for path in self._json_files(root):
            channel = self._channel_name(root, path)
            for message in self._read_messages(path):
                unit = self._message_unit(message, channel, path, root)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.edges.extend(self._thread_edges(result.units))
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

    def _channel_name(self, root: Path, path: Path) -> str:
        if root.is_file():
            return path.stem
        return root.name

    def _source_id(self, channel: str, ts: str) -> str:
        return f"slack_json:{channel}:{ts}"

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
                        "channel": channel,
                        "ts": ts,
                        "thread_ts": thread_ts,
                    },
                    created_at=unit.created_at,
                )
            )
        return edges

    def _edge_id(self, from_id: str, to_id: str) -> str:
        raw = "|".join([SourceProject.SLACK_JSON.value, EdgeRelation.REPLIES_TO.value, from_id, to_id])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"slack-json-replies-{digest[:16]}"

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
