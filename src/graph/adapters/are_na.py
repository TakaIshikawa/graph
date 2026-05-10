"""Adapter for Are.na JSON exports with channels and blocks."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class AreNaAdapter(SourceAdapter):
    """Import Are.na JSON exports preserving channels, blocks, and connections."""

    @property
    def name(self) -> str:
        return "are_na"

    @property
    def entity_types(self) -> list[str]:
        return ["block", "channel"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if not self.path:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        allowed_types = set(entity_types) if entity_types else None

        files = self._json_paths(root)
        for file_path in files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            self._process_data(data, result, allowed_types)

        result.units.sort(key=lambda u: (u.source_entity_type, u.source_id))
        result.edges.sort(key=lambda e: (e.from_unit_id, e.to_unit_id))
        return result

    def _json_paths(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root]
        return sorted(p for p in root.rglob("*.json") if p.is_file())

    def _process_data(
        self,
        data: Any,
        result: IngestResult,
        allowed_types: set[str] | None,
    ) -> None:
        if isinstance(data, dict):
            # Single channel object
            if "contents" in data or "title" in data:
                self._process_channel(data, result, allowed_types)
            # Collection of channels
            elif "channels" in data:
                for ch in data["channels"]:
                    if isinstance(ch, dict):
                        self._process_channel(ch, result, allowed_types)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    self._process_channel(item, result, allowed_types)

    def _process_channel(
        self,
        channel: dict[str, Any],
        result: IngestResult,
        allowed_types: set[str] | None,
    ) -> None:
        channel_title = channel.get("title") or channel.get("name") or ""
        channel_id_raw = str(channel.get("id") or channel.get("slug") or channel_title)

        # Create channel unit
        channel_source_id: str | None = None
        if not allowed_types or "channel" in allowed_types:
            digest = hashlib.sha1(channel_id_raw.encode("utf-8")).hexdigest()[:16]
            channel_source_id = f"are_na:channel:{digest}"

            metadata: dict[str, Any] = {}
            if channel.get("created_at"):
                metadata["created_at"] = channel["created_at"]
            if channel.get("updated_at"):
                metadata["updated_at"] = channel["updated_at"]

            unit = KnowledgeUnit(
                source_project=SourceProject.ARE_NA,
                source_id=channel_source_id,
                source_entity_type="channel",
                title=channel_title or "Untitled Channel",
                content=channel.get("description") or channel_title or "",
                content_type=ContentType.ARTIFACT,
                metadata=metadata,
                tags=[],
                created_at=self._parse_datetime(channel.get("created_at")) or datetime.now(timezone.utc),
                updated_at=self._parse_datetime(channel.get("updated_at")) or datetime.now(timezone.utc),
            )
            result.units.append(unit)

        # Process blocks (contents)
        blocks = channel.get("contents") or channel.get("blocks") or []
        for block in blocks:
            if not isinstance(block, dict):
                continue

            if allowed_types and "block" not in allowed_types:
                continue

            block_id_raw = str(block.get("id") or "")
            if not block_id_raw:
                continue

            block_title = block.get("title") or ""
            block_content = block.get("content") or block.get("description") or ""
            if not block_title and block_content:
                # Use first line of content as title
                block_title = block_content.split("\n", 1)[0][:120]

            digest = hashlib.sha1(block_id_raw.encode("utf-8")).hexdigest()[:16]
            block_source_id = f"are_na:block:{digest}"

            block_type = block.get("class") or block.get("type") or "Text"
            block_meta: dict[str, Any] = {
                "block_type": block_type,
            }
            if channel_title:
                block_meta["channel_title"] = channel_title
            if block.get("source") and isinstance(block["source"], dict):
                source_url = block["source"].get("url")
                if source_url:
                    block_meta["source_url"] = source_url
            elif block.get("source_url"):
                block_meta["source_url"] = block["source_url"]
            if block.get("created_at"):
                block_meta["created_at"] = block["created_at"]
            if block.get("updated_at"):
                block_meta["updated_at"] = block["updated_at"]

            tags: list[str] = []
            if channel_title:
                tags.append(channel_title.lower())

            unit = KnowledgeUnit(
                source_project=SourceProject.ARE_NA,
                source_id=block_source_id,
                source_entity_type="block",
                title=block_title or "Untitled Block",
                content=block_content or block_title or "",
                content_type=ContentType.ARTIFACT,
                metadata=block_meta,
                tags=sorted(tags),
                created_at=self._parse_datetime(block.get("created_at")) or datetime.now(timezone.utc),
                updated_at=self._parse_datetime(block.get("updated_at")) or datetime.now(timezone.utc),
            )
            result.units.append(unit)

            # Edge: channel contains block
            if channel_source_id:
                edge_id = self._edge_id(channel_source_id, block_source_id, "contains")
                result.edges.append(
                    KnowledgeEdge(
                        id=edge_id,
                        from_unit_id=channel_source_id,
                        to_unit_id=block_source_id,
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.ARE_NA.value,
                            "relation_type": "channel_membership",
                        },
                    )
                )

    def _edge_id(self, from_id: str, to_id: str, relation_type: str) -> str:
        raw = "|".join([SourceProject.ARE_NA.value, relation_type, from_id, to_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"are_na-{relation_type}-{digest}"

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        raw = str(value).strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (ValueError, TypeError):
            return None
