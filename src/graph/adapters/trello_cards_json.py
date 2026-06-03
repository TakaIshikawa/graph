"""Adapter for Trello card JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class TrelloCardsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "trello_cards_json"

    @property
    def entity_types(self) -> list[str]:
        return ["card"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "card" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                board = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            context = _context(board)
            for index, card in enumerate(_cards(board)):
                unit = self._unit(card, context, path.name, index)
                if unit and (not sync_at or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _unit(self, card: dict[str, Any], context: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        card_id = _text(card.get("id"))
        name = _text(card.get("name"))
        description = _text(card.get("desc") or card.get("description"))
        url = _text(card.get("url") or card.get("shortUrl"))
        if not any((card_id, name, description, url)):
            return None
        board_name = _text(context["board"].get("name") or card.get("boardName") or card.get("board"))
        list_id = _text(card.get("idList"))
        list_name = _text(_lookup(context["lists"], list_id).get("name") or card.get("listName") or card.get("list"))
        labels = _labels(card, context["labels"])
        members = [_member_name(_lookup(context["members"], member_id)) for member_id in card.get("idMembers", [])]
        members.extend(_member_name(member) for member in card.get("members", []) if isinstance(member, dict))
        members = [member for member in dict.fromkeys(members) if member]
        checklists = _checklists(card, context["checklists"])
        comments = _comments(card, context["members"])
        due_at = parse_datetime(card.get("due"))
        updated_at = parse_datetime(card.get("dateLastActivity") or card.get("updated_at") or card.get("updatedAt")) or due_at
        metadata = clean_metadata(
            {
                "card_id": card_id,
                "name": name,
                "description": description,
                "url": url,
                "board_name": board_name,
                "list_id": list_id,
                "list_name": list_name,
                "labels": labels,
                "members": members,
                "due": due_at.isoformat() if due_at else _text(card.get("due")),
                "closed": _bool(card.get("closed")),
                "checklists": checklists,
                "comments": comments,
                "source_file": source_file,
            }
        )
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.TRELLO_CARDS_JSON,
            source_id=f"trello_cards_json:{card_id}" if card_id else digest_source_id("trello_cards_json", name, url, source_file, index),
            source_entity_type="card",
            title=name or f"Trello card {card_id}",
            content=_content(name, description, metadata),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=[tag for tag in dict.fromkeys(["trello", "card", board_name, list_name, *labels]) if tag],
            created_at=updated_at or now,
            updated_at=updated_at or now,
        )


def _cards(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("cards", "items", "data", "results"):
            items = value.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return [value]
    return []


def _context(board: Any) -> dict[str, Any]:
    if not isinstance(board, dict):
        board = {}
    return {
        "board": board,
        "lists": {str(item.get("id")): item for item in board.get("lists", []) if isinstance(item, dict)},
        "labels": {str(item.get("id")): item for item in board.get("labels", []) if isinstance(item, dict)},
        "members": {str(item.get("id")): item for item in board.get("members", []) if isinstance(item, dict)},
        "checklists": {str(item.get("id")): item for item in board.get("checklists", []) if isinstance(item, dict)},
    }


def _lookup(items: dict[str, dict[str, Any]], item_id: Any) -> dict[str, Any]:
    return items.get(str(item_id), {}) if item_id is not None else {}


def _labels(card: dict[str, Any], labels_by_id: dict[str, dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for label in card.get("labels", []):
        if isinstance(label, dict):
            labels.append(_text(label.get("name") or label.get("color")))
        else:
            labels.append(_text(label))
    for label_id in card.get("idLabels", []):
        label = _lookup(labels_by_id, label_id)
        labels.append(_text(label.get("name") or label.get("color") or label_id))
    return [label for label in dict.fromkeys(labels) if label]


def _member_name(member: dict[str, Any]) -> str:
    return _text(member.get("fullName") or member.get("username") or member.get("name") or member.get("id"))


def _checklists(card: dict[str, Any], checklists_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    raw = [item for item in card.get("checklists", []) if isinstance(item, dict)]
    raw.extend(_lookup(checklists_by_id, checklist_id) for checklist_id in card.get("idChecklists", []))
    checklists: list[dict[str, Any]] = []
    for checklist in raw:
        items = []
        for item in checklist.get("checkItems", []):
            if isinstance(item, dict):
                items.append(clean_metadata({"id": _text(item.get("id")), "name": _text(item.get("name")), "state": _text(item.get("state"))}))
        checklists.append(clean_metadata({"id": _text(checklist.get("id")), "name": _text(checklist.get("name")), "items": items}))
    return [checklist for checklist in checklists if checklist]


def _comments(card: dict[str, Any], members_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    for action in card.get("actions", []):
        if not isinstance(action, dict) or _text(action.get("type")) != "commentCard":
            continue
        data = action.get("data") if isinstance(action.get("data"), dict) else {}
        member = action.get("memberCreator") if isinstance(action.get("memberCreator"), dict) else _lookup(members_by_id, action.get("idMemberCreator"))
        comments.append(clean_metadata({"id": _text(action.get("id")), "text": _text(data.get("text")), "member": _member_name(member), "date": _text(action.get("date"))}))
    return comments


def _content(name: str, description: str, metadata: dict[str, Any]) -> str:
    parts = [name, description]
    for key, label in (("board_name", "Board"), ("list_name", "List"), ("due", "Due"), ("url", "URL")):
        if metadata.get(key):
            parts.append(f"{label}: {metadata[key]}")
    for comment in metadata.get("comments", []):
        if comment.get("text"):
            parts.append(f"Comment: {comment['text']}")
    return "\n".join(part for part in parts if part)


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _text(value).casefold()
    if text in {"true", "yes", "1", "closed", "archived"}:
        return True
    if text in {"false", "no", "0", "open"}:
        return False
    return None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()
