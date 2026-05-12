"""Adapter for Miro board JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


SUPPORTED_TYPES = {
    "frame": "frame",
    "sticky_note": "sticky_note",
    "sticky": "sticky_note",
    "text": "text",
    "card": "card",
    "shape": "shape",
}


class MiroBoardJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "miro_board_json"

    @property
    def entity_types(self) -> list[str]:
        return ["frame", "sticky_note", "text", "card", "shape"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None

        for path in self._iter_paths():
            try:
                parsed = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            items = self._items(parsed)
            source_ids = {
                item_id: f"miro_board_json:{item_id}"
                for item in items
                if (item_id := self._item_id(item))
            }
            included_source_ids: set[str] = set()
            for item in items:
                entity_type = self._entity_type(item)
                if entity_type is None or entity_type not in allowed_types:
                    continue
                unit = self._unit_from_item(item, entity_type, path.name, source_ids)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
                included_source_ids.add(unit.source_id)
            result.edges.extend(self._frame_edges(items, source_ids, included_source_ids))
            result.edges.extend(self._connector_edges(items, source_ids, included_source_ids))

        result.units.sort(key=lambda unit: (unit.source_entity_type, unit.source_id))
        result.edges = sorted({edge.id: edge for edge in result.edges}.values(), key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.json") if path.is_file())

    def _items(self, parsed: Any) -> list[dict[str, Any]]:
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if not isinstance(parsed, dict):
            return []
        for key in ("items", "data", "widgets", "objects"):
            nested = parsed.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                child_items = nested.get("items") or nested.get("data")
                if isinstance(child_items, list):
                    return [item for item in child_items if isinstance(item, dict)]
        return [parsed] if self._entity_type(parsed) else []

    def _unit_from_item(
        self,
        item: dict[str, Any],
        entity_type: str,
        source_file: str,
        source_ids: dict[str, str],
    ) -> KnowledgeUnit | None:
        item_id = self._item_id(item)
        if not item_id:
            return None
        title = self._title(item, entity_type, item_id)
        content = self._content(item, title)
        created = self._parse_datetime(self._first(item, "createdAt", "created_at", "created"))
        updated = self._parse_datetime(self._first(item, "modifiedAt", "updatedAt", "updated_at", "modified_at", "lastModified"))
        now = datetime.now(timezone.utc)
        metadata = self._metadata(item, entity_type, item_id, source_file)
        return KnowledgeUnit(
            source_project=SourceProject.MIRO_BOARD_JSON,
            source_id=source_ids[item_id],
            source_entity_type=entity_type,
            title=title,
            content=content,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=list(dict.fromkeys(["miro", entity_type, *metadata.get("tags", [])])),
            created_at=created or updated or now,
            updated_at=updated or created or now,
        )

    def _metadata(
        self,
        item: dict[str, Any],
        entity_type: str,
        item_id: str,
        source_file: str,
    ) -> dict[str, Any]:
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        geometry = item.get("geometry") if isinstance(item.get("geometry"), dict) else {}
        position = self._position(item)
        dimensions = self._dimensions(item, geometry)
        parent_frame_id = self._parent_frame_id(item)
        metadata = {
            "item_id": item_id,
            "item_type": entity_type,
            "original_type": self._text(item.get("type")),
            "parent_frame_id": parent_frame_id,
            "position": position,
            "x": position.get("x"),
            "y": position.get("y"),
            "dimensions": dimensions,
            "width": dimensions.get("width"),
            "height": dimensions.get("height"),
            "rotation": self._number(item.get("rotation") or geometry.get("rotation")),
            "style": self._style(item),
            "links": self._links(item),
            "tags": self._tags(item),
            "creator": self._creator(item),
            "created_at": self._first(item, "createdAt", "created_at", "created"),
            "updated_at": self._first(item, "modifiedAt", "updatedAt", "updated_at", "modified_at", "lastModified"),
            "url": self._first(item, "url", "link", "selfLink"),
            "source_file": source_file,
            "text": self._text_value(item),
            "title": self._first(data, "title", "name"),
            "description": self._first(data, "description"),
            "item": item,
        }
        return {key: value for key, value in metadata.items() if value not in ("", None, [], {})}

    def _frame_edges(
        self,
        items: list[dict[str, Any]],
        source_ids: dict[str, str],
        included_source_ids: set[str],
    ) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        frame_source_ids = {
            self._item_id(item): source_ids[self._item_id(item)]
            for item in items
            if self._entity_type(item) == "frame" and self._item_id(item) in source_ids
        }
        for item in items:
            item_id = self._item_id(item)
            parent_frame_id = self._parent_frame_id(item)
            from_id = frame_source_ids.get(parent_frame_id)
            to_id = source_ids.get(item_id)
            if not from_id or not to_id or from_id == to_id:
                continue
            if from_id not in included_source_ids or to_id not in included_source_ids:
                continue
            edges.append(
                KnowledgeEdge(
                    id=self._edge_id(from_id, to_id, "frame_contains_item"),
                    from_unit_id=from_id,
                    to_unit_id=to_id,
                    relation=EdgeRelation.CONTAINS,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.MIRO_BOARD_JSON.value,
                        "relation_type": "miro_frame_contains_item",
                        "frame_id": parent_frame_id,
                        "item_id": item_id,
                    },
                )
            )
        return list({edge.id: edge for edge in edges}.values())

    def _connector_edges(
        self,
        items: list[dict[str, Any]],
        source_ids: dict[str, str],
        included_source_ids: set[str],
    ) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for item in items:
            connector_type = self._connector_type(item)
            if connector_type is None:
                continue
            start_id = self._connector_endpoint_id(item, "start")
            end_id = self._connector_endpoint_id(item, "end")
            from_id = source_ids.get(start_id)
            to_id = source_ids.get(end_id)
            if not from_id or not to_id or from_id == to_id:
                continue
            if from_id not in included_source_ids or to_id not in included_source_ids:
                continue
            connector_id = self._item_id(item)
            edges.append(
                KnowledgeEdge(
                    id=self._edge_id(from_id, to_id, f"{connector_type}_connects_items"),
                    from_unit_id=from_id,
                    to_unit_id=to_id,
                    relation=EdgeRelation.RELATES_TO,
                    source=EdgeSource.SOURCE,
                    metadata={
                        "source_project": SourceProject.MIRO_BOARD_JSON.value,
                        "relation_type": "miro_connector_connects_items",
                        "connector_type": connector_type,
                        "connector_id": connector_id,
                        "start_item_id": start_id,
                        "end_item_id": end_id,
                    },
                )
            )
        return edges

    def _entity_type(self, item: dict[str, Any]) -> str | None:
        raw_type = self._text(item.get("type") or item.get("itemType") or item.get("widgetType"))
        normalized = raw_type.replace("-", "_").replace(" ", "_").casefold()
        return SUPPORTED_TYPES.get(normalized)

    def _connector_type(self, item: dict[str, Any]) -> str | None:
        raw_type = self._text(item.get("type") or item.get("itemType") or item.get("widgetType"))
        normalized = raw_type.replace("-", "_").replace(" ", "_").casefold()
        if normalized in {"connector", "line", "arrow"}:
            return normalized
        return None

    def _connector_endpoint_id(self, item: dict[str, Any], side: str) -> str:
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        for container in (item, data):
            value = self._endpoint_value(container, side)
            if value:
                return value
        return ""

    def _endpoint_value(self, container: dict[str, Any], side: str) -> str:
        key_groups = {
            "start": (
                "startItem",
                "start_item",
                "startItemId",
                "start_item_id",
                "start",
                "from",
                "fromItem",
                "from_item",
                "fromItemId",
                "from_item_id",
                "source",
                "sourceItem",
                "source_item",
                "sourceItemId",
                "source_item_id",
            ),
            "end": (
                "endItem",
                "end_item",
                "endItemId",
                "end_item_id",
                "end",
                "to",
                "toItem",
                "to_item",
                "toItemId",
                "to_item_id",
                "target",
                "targetItem",
                "target_item",
                "targetItemId",
                "target_item_id",
            ),
        }
        for key in key_groups[side]:
            value = container.get(key)
            if isinstance(value, dict):
                endpoint_id = self._first(value, "id", "itemId", "item_id", "widgetId")
            else:
                endpoint_id = self._text(value)
            if endpoint_id:
                return endpoint_id
        return ""

    def _item_id(self, item: dict[str, Any]) -> str:
        return self._first(item, "id", "item_id", "widgetId")

    def _parent_frame_id(self, item: dict[str, Any]) -> str:
        parent = item.get("parent") if isinstance(item.get("parent"), dict) else {}
        return self._first(item, "parentFrameId", "parent_frame_id", "frameId") or self._first(parent, "id")

    def _position(self, item: dict[str, Any]) -> dict[str, float]:
        position = item.get("position") if isinstance(item.get("position"), dict) else {}
        return {
            key: value
            for key, value in {
                "x": self._number(item.get("x") if item.get("x") is not None else position.get("x")),
                "y": self._number(item.get("y") if item.get("y") is not None else position.get("y")),
                "origin": self._text(position.get("origin")),
            }.items()
            if value not in ("", None)
        }

    def _dimensions(self, item: dict[str, Any], geometry: dict[str, Any]) -> dict[str, float]:
        return {
            key: value
            for key, value in {
                "width": self._number(item.get("width") if item.get("width") is not None else geometry.get("width")),
                "height": self._number(item.get("height") if item.get("height") is not None else geometry.get("height")),
            }.items()
            if value is not None
        }

    def _style(self, item: dict[str, Any]) -> dict[str, Any]:
        style = item.get("style")
        if isinstance(style, dict):
            return dict(style)
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        data_style = data.get("style")
        return dict(data_style) if isinstance(data_style, dict) else {}

    def _links(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        links: list[dict[str, Any]] = []
        for key in ("links", "attachedLinks"):
            raw_links = item.get(key)
            if isinstance(raw_links, list):
                for link in raw_links:
                    if isinstance(link, dict):
                        links.append(dict(link))
                    elif self._text(link):
                        links.append({"url": self._text(link)})
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        for key in ("url", "link"):
            value = self._text(data.get(key) or item.get(key))
            if value:
                links.append({"url": value})
        return links

    def _tags(self, item: dict[str, Any]) -> list[str]:
        raw_tags = item.get("tags")
        tags: list[str] = []
        if isinstance(raw_tags, list):
            for tag in raw_tags:
                if isinstance(tag, dict):
                    label = self._first(tag, "title", "name")
                else:
                    label = self._text(tag)
                if label and label not in tags:
                    tags.append(label)
        return tags

    def _creator(self, item: dict[str, Any]) -> dict[str, Any]:
        for key in ("createdBy", "creator", "created_by", "user"):
            creator = item.get(key)
            if isinstance(creator, dict):
                return dict(creator)
            if self._text(creator):
                return {"name": self._text(creator)}
        return {}

    def _text_value(self, item: dict[str, Any]) -> str:
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        return self._first(data, "content", "text", "plainText", "title", "name") or self._first(
            item, "content", "text", "title", "name"
        )

    def _title(self, item: dict[str, Any], entity_type: str, item_id: str) -> str:
        text = self._text_value(item)
        if text:
            return text.splitlines()[0][:120]
        return f"Miro {entity_type.replace('_', ' ')} {item_id}"

    def _content(self, item: dict[str, Any], title: str) -> str:
        parts = [self._text_value(item) or title]
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        description = self._first(data, "description")
        if description:
            parts.append(description)
        url = self._first(item, "url", "link", "selfLink") or self._first(data, "url", "link")
        if url:
            parts.append(f"URL: {url}")
        return "\n".join(part for part in parts if part)

    def _edge_id(self, from_id: str, to_id: str, relation_type: str) -> str:
        digest = hashlib.sha256("|".join([from_id, to_id, relation_type]).encode("utf-8")).hexdigest()[:24]
        return f"miro-board-json-{relation_type}-{digest}"

    def _first(self, row: dict[str, Any], *keys: str) -> str:
        lowered = {str(key).strip().casefold(): value for key, value in row.items()}
        compact = {self._compact_key(key): value for key, value in row.items()}
        for key in keys:
            value = row.get(key)
            if value is None:
                value = lowered.get(key.casefold())
            if value is None:
                value = compact.get(self._compact_key(key))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _compact_key(self, value: Any) -> str:
        return "".join(ch for ch in str(value).casefold() if ch.isalnum())

    def _text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _number(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            parsed = float(str(value).strip())
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
        return parsed

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        text = str(value).strip()
        for candidate in (text, f"{text}T00:00:00"):
            try:
                return self._ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
            except ValueError:
                pass
        return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
