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
        return ["card", "check_item", "list", "label", "member"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        requested_types = set(entity_types) if entity_types is not None else {"card", "check_item", "list"}
        include_cards = "card" in requested_types
        include_check_items = "check_item" in requested_types
        include_lists = "list" in requested_types
        include_labels = "label" in requested_types
        include_members = "member" in requested_types
        if not include_cards and not include_check_items and not include_lists and not include_labels and not include_members:
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        card_units: list[KnowledgeUnit] = []
        cards_by_list: dict[str, list[KnowledgeUnit]] = {}
        raw_cards_by_list: dict[str, list[dict[str, Any]]] = {}
        list_records: dict[str, dict[str, Any]] = {}
        for path in self._iter_paths():
            try:
                board = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(board, dict):
                continue
            context = self._context(board)
            for board_list in board.get("lists", []):
                if not isinstance(board_list, dict):
                    continue
                list_id = self._text(board_list.get("id"))
                list_name = self._lookup_name(board_list)
                if not list_id and not list_name:
                    continue
                key = self._list_key(list_id, list_name)
                list_records.setdefault(key, {"list": board_list, "source_files": set()})
                list_records[key]["source_files"].add(path.name)
            for card in board.get("cards", []):
                if not isinstance(card, dict):
                    continue
                list_id = self._text(card.get("idList"))
                list_name = self._lookup_name(context["lists"].get(list_id))
                if not list_id and not list_name:
                    continue
                raw_cards_by_list.setdefault(self._list_key(list_id, list_name), []).append(card)
            for card in board.get("cards", []):
                if not isinstance(card, dict):
                    continue
                unit = self._unit_from_card(card, context, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                card_units.append(unit)
                list_id = self._text(card.get("idList"))
                list_name = self._lookup_name(context["lists"].get(list_id))
                if list_id or list_name:
                    cards_by_list.setdefault(self._list_key(list_id, list_name), []).append(unit)
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
        list_units = self._list_units(list_records, raw_cards_by_list, cards_by_list) if include_lists else []
        if include_lists:
            result.units.extend(list_units)
        if include_lists and include_cards:
            result.edges.extend(self._list_card_edges(list_units, cards_by_list))
        label_units = self._aggregate_units("label", card_units) if include_labels else []
        member_units = self._aggregate_units("member", card_units) if include_members else []
        result.units.extend(label_units)
        result.units.extend(member_units)
        if include_cards:
            result.edges.extend(self._aggregate_card_edges(label_units, card_units, "label"))
            result.edges.extend(self._aggregate_card_edges(member_units, card_units, "member"))
        result.units.sort(key=lambda unit: unit.source_id)
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
        checklist_items = self._checklist_items(card.get("idChecklists", []), context["checklists"])
        metadata = {
            "card_id": card_id,
            "name": name,
            "description": self._text(card.get("desc")),
            "due": self._text(card.get("due")),
            "closed": bool(card.get("closed")),
            "list_id": self._text(card.get("idList")),
            "list_name": list_name,
            "labels": labels,
            "members": [item for item in members if item],
            "checklists": checklists,
            "checklist_items": checklist_items,
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

    def _list_units(
        self,
        list_records: dict[str, dict[str, Any]],
        raw_cards_by_list: dict[str, list[dict[str, Any]]],
        cards_by_list: dict[str, list[KnowledgeUnit]],
    ) -> list[KnowledgeUnit]:
        units: list[KnowledgeUnit] = []
        now = datetime.now(timezone.utc)
        for key, info in list_records.items():
            board_list = info["list"]
            list_id = self._text(board_list.get("id"))
            name = self._lookup_name(board_list) or key.removeprefix("name:")
            raw_cards = raw_cards_by_list.get(key, [])
            linked_cards = cards_by_list.get(key, [])
            open_count = len([card for card in raw_cards if not bool(card.get("closed"))])
            closed_count = len([card for card in raw_cards if bool(card.get("closed"))])
            metadata = {
                "list_id": list_id,
                "name": name,
                "closed": bool(board_list.get("closed")),
                "position": board_list.get("pos"),
                "card_count": len(raw_cards),
                "open_count": open_count,
                "closed_count": closed_count,
                "card_source_ids": sorted({unit.source_id for unit in linked_cards}),
                "source_files": sorted(info["source_files"]),
                "source_file": sorted(info["source_files"])[0] if len(info["source_files"]) == 1 else "",
            }
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.TRELLO_BOARD_JSON,
                    source_id=self._list_source_id(list_id, name),
                    source_entity_type="list",
                    title=name or f"Trello list {list_id}",
                    content=self._list_content(metadata),
                    content_type=ContentType.METADATA,
                    metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
                    tags=["trello", "list"],
                    created_at=now,
                    updated_at=now,
                )
            )
        return units

    def _list_card_edges(
        self,
        list_units: list[KnowledgeUnit],
        cards_by_list: dict[str, list[KnowledgeUnit]],
    ) -> list[KnowledgeEdge]:
        list_ids = {self._list_key(self._text(unit.metadata.get("list_id")), self._text(unit.metadata.get("name"))): unit.source_id for unit in list_units}
        edges: list[KnowledgeEdge] = []
        for key, cards in cards_by_list.items():
            list_unit_id = list_ids.get(key)
            if not list_unit_id:
                continue
            for card in cards:
                edges.append(
                    KnowledgeEdge(
                        id=self._source_edge_id(list_unit_id, card.source_id, "list_card"),
                        from_unit_id=list_unit_id,
                        to_unit_id=card.source_id,
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "kind": "list_card",
                            "relation_type": "trello_list_card",
                            "list_id": card.metadata.get("list_id", ""),
                            "card_id": card.metadata.get("card_id", ""),
                        },
                    )
                )
        return list({edge.id: edge for edge in edges}.values())

    def _aggregate_units(self, entity_type: str, cards: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
        metadata_key = "labels" if entity_type == "label" else "members"
        grouped: dict[str, list[KnowledgeUnit]] = {}
        for card in cards:
            for value in card.metadata.get(metadata_key, []):
                name = self._text(value)
                if name:
                    grouped.setdefault(name, []).append(card)

        units: list[KnowledgeUnit] = []
        now = datetime.now(timezone.utc)
        for name, linked_cards in grouped.items():
            created_at = min((card.created_at for card in linked_cards), default=now)
            updated_at = max((card.updated_at for card in linked_cards), default=created_at)
            card_source_ids = sorted({card.source_id for card in linked_cards})
            metadata = {
                "name": name,
                "card_source_ids": card_source_ids,
                "card_count": len(card_source_ids),
                "lists": sorted({list_name for card in linked_cards if (list_name := self._text(card.metadata.get("list_name")))}),
                "latest_updated_at": updated_at.isoformat(),
            }
            units.append(
                KnowledgeUnit(
                    source_project=SourceProject.TRELLO_BOARD_JSON,
                    source_id=self._aggregate_source_id(entity_type, name),
                    source_entity_type=entity_type,
                    title=f"Trello {entity_type}: {name}",
                    content=f"Trello {entity_type}: {name}\nCards: {len(card_source_ids)}",
                    content_type=ContentType.METADATA,
                    metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
                    tags=["trello", entity_type],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return units

    def _aggregate_card_edges(self, aggregate_units: list[KnowledgeUnit], cards: list[KnowledgeUnit], entity_type: str) -> list[KnowledgeEdge]:
        metadata_key = "labels" if entity_type == "label" else "members"
        aggregate_ids = {unit.metadata["name"]: unit.source_id for unit in aggregate_units}
        edges: list[KnowledgeEdge] = []
        for card in cards:
            for value in card.metadata.get(metadata_key, []):
                name = self._text(value)
                target_id = aggregate_ids.get(name)
                if target_id:
                    edges.append(self._relation_edge(card.source_id, target_id, f"card_{entity_type}", entity_type, name))
        return edges

    def _list_content(self, metadata: dict[str, Any]) -> str:
        parts = [metadata.get("name", "")]
        parts.append(f"Cards: {metadata.get('card_count', 0)}")
        parts.append(f"Open: {metadata.get('open_count', 0)}")
        parts.append(f"Closed: {metadata.get('closed_count', 0)}")
        return "\n".join(part for part in parts if part)

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

    def _checklist_items(self, checklist_ids: list[Any], known_checklists: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for checklist in self._card_checklists(checklist_ids, known_checklists):
            checklist_id = self._text(checklist.get("id"))
            checklist_name = self._lookup_name(checklist)
            check_items = checklist.get("checkItems", [])
            if not isinstance(check_items, list):
                continue
            for index, check_item in enumerate(check_items):
                if not isinstance(check_item, dict):
                    continue
                item_name = self._text(check_item.get("name"))
                item_id = self._text(check_item.get("id"))
                if not item_name and not item_id:
                    continue
                item = {
                    "checklist_id": checklist_id,
                    "checklist_name": checklist_name,
                    "item_id": item_id,
                    "item_name": item_name,
                    "state": self._text(check_item.get("state")),
                    "complete": self._text(check_item.get("state")).casefold() == "complete",
                    "due": self._text(check_item.get("due")),
                    "position": check_item.get("pos", index),
                }
                items.append({key: value for key, value in item.items() if value not in ("", None, [])})
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
        for item in metadata.get("checklist_items", []):
            if not isinstance(item, dict):
                continue
            line = item.get("item_name") or item.get("item_id")
            if not line:
                continue
            state = item.get("state")
            checklist = item.get("checklist_name")
            due = item.get("due")
            details = [value for value in (state, f"Checklist: {checklist}" if checklist else "", f"Due: {due}" if due else "") if value]
            parts.append(f"Checklist item: {line}" + (f" ({'; '.join(details)})" if details else ""))
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

    def _list_key(self, list_id: str, name: str) -> str:
        return f"id:{list_id}" if list_id else f"name:{name.casefold()}"

    def _list_source_id(self, list_id: str, name: str) -> str:
        if list_id:
            return f"trello_board_json:list:{list_id}"
        return self._source_id("list", name.casefold())

    def _aggregate_source_id(self, entity_type: str, name: str) -> str:
        return self._source_id(entity_type, name.casefold())

    def _relation_edge(self, source_id: str, target_id: str, relation_type: str, kind: str, value: str) -> KnowledgeEdge:
        return KnowledgeEdge(
            id=self._source_edge_id(source_id, target_id, relation_type),
            from_unit_id=source_id,
            to_unit_id=target_id,
            relation=EdgeRelation.RELATES_TO,
            source=EdgeSource.SOURCE,
            metadata={"kind": kind, "value": value, "relation_type": f"trello_{relation_type}"},
        )

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
