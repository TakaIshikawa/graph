"""Adapter for Trello board JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class TrelloBoardJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "trello_board_json"

    @property
    def entity_types(self) -> list[str]:
        return ["card", "check_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        requested_types = set(entity_types or self.entity_types)
        include_cards = "card" in requested_types
        include_check_items = "check_item" in requested_types
        if not include_cards and not include_check_items:
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                board = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(board, dict):
                continue
            context = self._context(board)
            for card in board.get("cards", []):
                if not isinstance(card, dict):
                    continue
                unit = self._unit_from_card(card, context, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                check_item_units = self._check_item_units(card, context, path.name, unit)
                if include_cards:
                    result.units.append(unit)
                    result.edges.extend(self._edges_for_unit(unit))
                if include_check_items:
                    result.units.extend(check_item_units)
                if include_cards and include_check_items:
                    result.edges.extend(
                        self._check_item_edge(unit, check_item_unit) for check_item_unit in check_item_units
                    )
        result.units.sort(key=lambda unit: unit.source_id)
        result.edges.sort(key=lambda edge: edge.id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.json") if child.is_file())

    def _context(self, board: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            "lists": {str(item.get("id")): item for item in board.get("lists", []) if isinstance(item, dict)},
            "labels": {str(item.get("id")): item for item in board.get("labels", []) if isinstance(item, dict)},
            "members": {str(item.get("id")): item for item in board.get("members", []) if isinstance(item, dict)},
            "checklists": {str(item.get("id")): item for item in board.get("checklists", []) if isinstance(item, dict)},
        }

    def _unit_from_card(self, card: dict[str, Any], context: dict[str, dict[str, Any]], source_file: str) -> KnowledgeUnit | None:
        card_id = self._text(card.get("id"))
        name = self._text(card.get("name"))
        if not card_id and not name:
            return None
        created = self._parse_datetime(card.get("dateLastActivity")) or self._parse_datetime(card.get("due"))
        updated = self._parse_datetime(card.get("dateLastActivity")) or created
        list_name = self._lookup_name(context["lists"].get(self._text(card.get("idList"))))
        labels = self._labels(card, context["labels"])
        members = [self._lookup_member(context["members"].get(str(member_id))) for member_id in card.get("idMembers", [])]
        checklists = self._checklists(card.get("idChecklists", []), context["checklists"])
        metadata = {
            "card_id": card_id,
            "name": name,
            "description": self._text(card.get("desc")),
            "due": self._text(card.get("due")),
            "closed": bool(card.get("closed")),
            "list_name": list_name,
            "labels": labels,
            "members": [item for item in members if item],
            "checklists": checklists,
            "url": self._text(card.get("url") or card.get("shortUrl")),
            "source_file": source_file,
            "card": card,
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.TRELLO_BOARD_JSON,
            source_id=f"trello_board_json:{card_id}" if card_id else self._source_id(name, metadata["url"]),
            source_entity_type="card",
            title=name or f"Trello card {card_id}",
            content=self._content(name, metadata),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(["trello", "card", *labels])),
            created_at=created or now,
            updated_at=updated or created or now,
        )

    def _check_item_units(
        self,
        card: dict[str, Any],
        context: dict[str, dict[str, Any]],
        source_file: str,
        card_unit: KnowledgeUnit,
    ) -> list[KnowledgeUnit]:
        card_id = self._text(card.get("id"))
        card_name = self._text(card.get("name"))
        card_url = self._text(card.get("url") or card.get("shortUrl"))
        list_name = card_unit.metadata.get("list_name", "")
        labels = card_unit.metadata.get("labels", [])
        created = card_unit.created_at
        updated = card_unit.updated_at
        units: list[KnowledgeUnit] = []

        for checklist in self._card_checklists(card.get("idChecklists", []), context["checklists"]):
            checklist_id = self._text(checklist.get("id"))
            checklist_name = self._lookup_name(checklist)
            check_items = checklist.get("checkItems", [])
            if not isinstance(check_items, list):
                continue
            for index, check_item in enumerate(check_items):
                if not isinstance(check_item, dict):
                    continue
                item_id = self._text(check_item.get("id"))
                item_name = self._text(check_item.get("name"))
                if not item_id and not item_name:
                    continue
                member_ids = [self._text(member_id) for member_id in check_item.get("idMembers", [])]
                metadata = {
                    "checklist_id": checklist_id,
                    "checklist_name": checklist_name,
                    "item_id": item_id,
                    "item_name": item_name,
                    "state": self._text(check_item.get("state")),
                    "due": self._text(check_item.get("due")),
                    "dueReminder": self._text(check_item.get("dueReminder")),
                    "member_ids": [member_id for member_id in member_ids if member_id],
                    "card_id": card_id,
                    "card_name": card_name,
                    "card_url": card_url,
                    "list_name": list_name,
                    "labels": labels,
                    "source_file": source_file,
                }
                units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.TRELLO_BOARD_JSON,
                        source_id=self._check_item_source_id(card_id, checklist_id, item_id, item_name, index),
                        source_entity_type="check_item",
                        title=item_name or f"Trello checklist item {item_id}",
                        content=self._check_item_content(metadata),
                        content_type=ContentType.ARTIFACT,
                        metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
                        tags=list(dict.fromkeys(["trello", "check_item", *labels])),
                        created_at=created,
                        updated_at=updated,
                    )
                )
        return units

    def _edges_for_unit(self, unit: KnowledgeUnit) -> list[KnowledgeEdge]:
        edges: list[KnowledgeEdge] = []
        for kind, values in (
            ("list", [unit.metadata.get("list_name")]),
            ("label", unit.metadata.get("labels", [])),
            ("member", unit.metadata.get("members", [])),
            ("checklist", [item.get("name") for item in unit.metadata.get("checklists", []) if isinstance(item, dict)]),
        ):
            for value in values:
                if value:
                    edges.append(self._edge(unit.source_id, f"trello:{kind}:{value}", EdgeRelation.RELATES_TO, kind, str(value)))
        return edges

    def _labels(self, card: dict[str, Any], known_labels: dict[str, Any]) -> list[str]:
        labels: list[str] = []
        for label in card.get("labels", []):
            name = self._lookup_name(label if isinstance(label, dict) else known_labels.get(str(label)))
            if name and name not in labels:
                labels.append(name)
        for label_id in card.get("idLabels", []):
            name = self._lookup_name(known_labels.get(str(label_id)))
            if name and name not in labels:
                labels.append(name)
        return labels

    def _checklists(self, checklist_ids: list[Any], known_checklists: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for checklist in self._card_checklists(checklist_ids, known_checklists):
            checks = checklist.get("checkItems", [])
            items.append(
                {
                    "name": self._lookup_name(checklist),
                    "total": len(checks) if isinstance(checks, list) else 0,
                    "complete": len([item for item in checks if isinstance(item, dict) and item.get("state") == "complete"]) if isinstance(checks, list) else 0,
                }
            )
        return items

    def _card_checklists(self, checklist_ids: list[Any], known_checklists: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(checklist_ids, list):
            return []
        checklists: list[dict[str, Any]] = []
        for checklist_id in checklist_ids:
            checklist = known_checklists.get(str(checklist_id))
            if isinstance(checklist, dict):
                checklists.append(checklist)
        return checklists

    def _content(self, name: str, metadata: dict[str, Any]) -> str:
        parts = [name, metadata.get("description", "")]
        for key, label in (("list_name", "List"), ("due", "Due"), ("url", "URL")):
            if metadata.get(key):
                parts.append(f"{label}: {metadata[key]}")
        if metadata.get("labels"):
            parts.append(f"Labels: {', '.join(metadata['labels'])}")
        if metadata.get("members"):
            parts.append(f"Members: {', '.join(metadata['members'])}")
        return "\n".join(item for item in parts if item)

    def _check_item_content(self, metadata: dict[str, Any]) -> str:
        parts = [metadata.get("item_name", "")]
        for key, label in (
            ("state", "State"),
            ("checklist_name", "Checklist"),
            ("card_name", "Card"),
            ("list_name", "List"),
            ("due", "Due"),
            ("card_url", "URL"),
        ):
            if metadata.get(key):
                parts.append(f"{label}: {metadata[key]}")
        if metadata.get("labels"):
            parts.append(f"Labels: {', '.join(metadata['labels'])}")
        if metadata.get("member_ids"):
            parts.append(f"Member IDs: {', '.join(metadata['member_ids'])}")
        return "\n".join(item for item in parts if item)

    def _check_item_edge(self, card_unit: KnowledgeUnit, check_item_unit: KnowledgeUnit) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._source_edge_id(card_unit.source_id, check_item_unit.source_id, "check_item"),
            from_unit_id=card_unit.source_id,
            to_unit_id=check_item_unit.source_id,
            relation=EdgeRelation.CONTAINS,
            source=EdgeSource.SOURCE,
            metadata={
                "kind": "check_item",
                "value": check_item_unit.metadata.get("item_name", ""),
                "relation_type": "trello_card_check_item",
                "card_id": card_unit.metadata.get("card_id", ""),
                "checklist_id": check_item_unit.metadata.get("checklist_id", ""),
                "item_id": check_item_unit.metadata.get("item_id", ""),
            },
        )

    def _edge(self, source_id: str, target: str, relation: EdgeRelation, kind: str, value: str) -> KnowledgeEdge:
        digest = hashlib.sha256(f"{source_id}|{relation}|{target}".encode("utf-8")).hexdigest()[:24]
        return KnowledgeEdge(id=f"trello_board_json:{digest}", from_unit_id=source_id, to_unit_id=target, relation=relation, source=EdgeSource.SOURCE, metadata={"kind": kind, "value": value})

    def _source_edge_id(self, source_id: str, target: str, kind: str) -> str:
        digest = hashlib.sha256(f"{source_id}|{kind}|{target}".encode("utf-8")).hexdigest()[:24]
        return f"trello_board_json:{digest}"

    def _lookup_name(self, item: Any) -> str:
        return self._text(item.get("name") if isinstance(item, dict) else item)

    def _lookup_member(self, item: Any) -> str:
        if not isinstance(item, dict):
            return self._text(item)
        return self._text(item.get("fullName") or item.get("username") or item.get("name"))

    def _source_id(self, *parts: str) -> str:
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
        return f"trello_board_json:{digest}"

    def _check_item_source_id(
        self,
        card_id: str,
        checklist_id: str,
        item_id: str,
        item_name: str,
        index: int,
    ) -> str:
        if card_id and checklist_id and item_id:
            return f"trello_board_json:{card_id}:check_item:{checklist_id}:{item_id}"
        return self._source_id(card_id, checklist_id, item_id, item_name, str(index))

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = self._text(value)
        if not text:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
