"""Adapter for Linear comments JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class LinearCommentsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "linear_comments_json"

    @property
    def entity_types(self) -> list[str]:
        return ["comment"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "comment" not in set(entity_types or self.entity_types):
            return result
        sync_at = _ensure_utc(since.last_sync_at) if since else None
        for path in _iter_json(self.path):
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for item in _items(data):
                unit = self._unit(item, path.name)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, item: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        body = _text(item.get("body") or item.get("text") or item.get("content"))
        if not body and item.get("deletedAt"):
            body = "[deleted comment]"
        if not body:
            return None
        issue = item.get("issue") if isinstance(item.get("issue"), dict) else {}
        user = item.get("user") or item.get("author")
        created = _parse_dt(item.get("createdAt") or item.get("created_at"))
        updated = _parse_dt(item.get("updatedAt") or item.get("updated_at")) or created or datetime.now(timezone.utc)
        issue_identifier = _text(issue.get("identifier") or item.get("issueIdentifier"))
        comment_id = _text(item.get("id")) or _digest(body, issue_identifier)
        metadata = _clean(
            {
                "comment_id": comment_id,
                "url": _text(item.get("url")),
                "created_at": created.isoformat() if created else "",
                "updated_at": updated.isoformat(),
                "user": _user(user),
                "issue_id": _text(issue.get("id") or item.get("issueId")),
                "issue_identifier": issue_identifier,
                "issue_title": _text(issue.get("title") or item.get("issueTitle")),
                "parent_id": _text(item.get("parentId") or (item.get("parent", {}).get("id") if isinstance(item.get("parent"), dict) else "")),
                "thread_id": _text(item.get("threadId") or item.get("thread_id")),
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=self.name,
            source_id=f"{self.name}:{_digest(comment_id)}",
            source_entity_type="comment",
            title=f"Linear comment on {issue_identifier}" if issue_identifier else f"Linear comment {comment_id}",
            content=body,
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["linear", "comment"],
            created_at=created or updated,
            updated_at=updated,
        )


def _items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        nested = data.get("comments") or data.get("nodes") or data.get("data")
        return _items(nested) if nested is not None else [data]
    return []


def _user(value: Any) -> str:
    return _text(value.get("name") or value.get("displayName") or value.get("email") or value.get("id")) if isinstance(value, dict) else _text(value)


def _iter_json(path: str) -> list[Path]:
    root = Path(path).expanduser()
    if root.is_file() and root.suffix.lower() == ".json":
        return [root]
    return sorted(root.rglob("*.json")) if root.is_dir() else []


def _parse_dt(value: Any) -> datetime | None:
    try:
        return _ensure_utc(datetime.fromisoformat(_text(value).replace("Z", "+00:00")))
    except ValueError:
        return None


def _ensure_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _clean(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if value not in ("", None, [])}


def _digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()[:24]
