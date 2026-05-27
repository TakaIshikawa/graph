"""Adapter for GitHub Discussions JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class GithubDiscussionsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_discussions_json"

    @property
    def entity_types(self) -> list[str]:
        return ["discussion"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "discussion" not in set(entity_types or self.entity_types):
            return result
        sync_at = _ensure_utc(since.last_sync_at) if since else None
        for path in _iter_json(self.path):
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for item in _nodes(data, "discussions"):
                unit = self._unit(item, path.name)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, item: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        title = _text(item.get("title"))
        body = _text(item.get("bodyText") or item.get("body"))
        number = _text(item.get("number") or item.get("id"))
        if not title and not body:
            return None
        created = _parse_dt(item.get("createdAt"))
        updated = _parse_dt(item.get("updatedAt")) or created or datetime.now(timezone.utc)
        answer = item.get("answer") if isinstance(item.get("answer"), dict) else item.get("acceptedAnswer")
        answer_meta = answer if isinstance(answer, dict) else {}
        labels = _names(item.get("labels"))
        category = item.get("category") if isinstance(item.get("category"), dict) else {}
        metadata = _clean(
            {
                "number": number,
                "url": _text(item.get("url")),
                "category": _text(category.get("name") if isinstance(category, dict) else item.get("category")),
                "author": _login(item.get("author")),
                "created_at": created.isoformat() if created else "",
                "updated_at": updated.isoformat(),
                "answered": bool(item.get("isAnswered") or answer_meta),
                "accepted_answer_id": _text(answer_meta.get("id")),
                "accepted_answer_url": _text(answer_meta.get("url")),
                "labels": labels,
                "comment_count": _count(item.get("comments")),
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=self.name,
            source_id=f"{self.name}:{_digest(number or title)}",
            source_entity_type="discussion",
            title=title or f"Discussion {number}",
            content="\n\n".join(part for part in (title, body) if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["github", "discussion", *labels],
            created_at=created or updated,
            updated_at=updated,
        )


def _nodes(value: Any, preferred_key: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [node for item in value for node in _nodes(item, preferred_key)]
    if not isinstance(value, dict):
        return []
    if isinstance(value.get("node"), dict):
        return _nodes(value["node"], preferred_key)
    for key in (preferred_key, "nodes", "edges", "data", "repository"):
        if key in value:
            found = _nodes(value[key], preferred_key)
            if found:
                return found
    return [value] if value.get("title") or value.get("body") or value.get("bodyText") else []


def _names(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.get("nodes") or value.get("edges") or []
    if not isinstance(value, list):
        return []
    return [name for name in dict.fromkeys(_text((item.get("node") if isinstance(item, dict) and isinstance(item.get("node"), dict) else item).get("name") if isinstance(item, dict) else item) for item in value) if name]


def _login(value: Any) -> str:
    return _text(value.get("login") or value.get("name")) if isinstance(value, dict) else _text(value)


def _count(value: Any) -> int:
    if isinstance(value, dict):
        return int(value.get("totalCount") or len(value.get("nodes") or []))
    return len(value) if isinstance(value, list) else 0


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
