"""Adapter for Mastodon bookmarked status JSON exports."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class MastodonBookmarksJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "mastodon_bookmarks_json"

    @property
    def entity_types(self) -> list[str]:
        return ["status"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "status" not in entity_types:
            return result
        for path in self._paths():
            try:
                records = self._records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit(record, path.name, index)
                if unit:
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _paths(self) -> list[Path]:
        root = Path(self.path).expanduser() if self.path else None
        if root is None:
            return []
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        return sorted(root.rglob("*.json")) if root.is_dir() else []

    def _records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("bookmarks", "statuses", "orderedItems", "items"):
                if isinstance(parsed.get(key), list):
                    return [item for item in parsed[key] if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        status = record.get("status") if isinstance(record.get("status"), dict) else record
        status_id = _first(status, "id", "uri")
        url = _first(status, "url", "uri")
        account = status.get("account") if isinstance(status.get("account"), dict) else {}
        author = _first(account, "display_name", "displayName", "acct", "username")
        content = _html_text(_first(status, "content", "text"))
        created_text = _first(status, "created_at", "createdAt", "published")
        tags = _tags(status.get("tags"))
        language = _first(status, "language", "lang")
        if not any([status_id, url, content]):
            return None
        created_at = parse_datetime(created_text) or datetime.now(timezone.utc)
        metadata = clean_metadata({"status_id": status_id, "url": url, "external_url": url, "author": author, "acct": _first(account, "acct", "username"), "created_at": created_text, "language": language, "tags": tags, "reblogs_count": _number(status.get("reblogs_count")), "favourites_count": _number(status.get("favourites_count")), "replies_count": _number(status.get("replies_count")), "source_file": source_file})
        title = f"{author}: {content[:60]}" if author and content else (content[:80] or url or status_id)
        return KnowledgeUnit(source_project="mastodon_bookmarks_json", source_id=digest_source_id("mastodon_bookmarks_json", status_id or url or index), source_entity_type="status", title=title, content=_content(author, content, url, tags), content_type=ContentType.ARTIFACT, metadata=metadata, tags=tags, created_at=created_at, updated_at=created_at)


def _first(row: dict[str, Any], *keys: str) -> str:
    lowered = {str(key).casefold(): value for key, value in row.items()}
    for key in keys:
        value = row.get(key, lowered.get(key.casefold()))
        if value is not None and not isinstance(value, (dict, list)) and str(value).strip():
            return str(value).strip()
    return ""


def _html_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(value))).strip()


def _tags(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    for item in value:
        tag = _first(item, "name") if isinstance(item, dict) else str(item).strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _number(value: object) -> int | None:
    try:
        return int(value) if value is not None and str(value).strip() else None
    except ValueError:
        return None


def _content(author: str, content: str, url: str, tags: list[str]) -> str:
    parts = [f"Author: {author}" if author else "", content, f"URL: {url}" if url else "", f"Tags: {', '.join(tags)}" if tags else ""]
    return "\n".join(part for part in parts if part)
