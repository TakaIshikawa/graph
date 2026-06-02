"""Adapter for Mastodon favourite status JSON exports."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, iter_paths, parse_datetime
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class MastodonFavouritesJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "mastodon_favourites_json"

    @property
    def entity_types(self) -> list[str]:
        return ["favourite", "status"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types) if entity_types is not None else {"favourite"}
        if not allowed_types.intersection(self.entity_types):
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".json"}):
            try:
                records = self._records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for index, record in enumerate(records):
                unit = self._unit(record, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                if "favourite" in allowed_types or "status" in allowed_types:
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        return list(_iter_status_records(parsed))

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        status = _status_record(record)
        status_id = _first(status, "id", "uri")
        url = _first(status, "url", "uri")
        content = _html_text(_first(status, "content", "text", "summary"))
        account = status.get("account") if isinstance(status.get("account"), dict) else {}
        author = _first(account, "display_name", "displayName", "acct", "username")
        acct = _first(account, "acct", "username")
        created_text = _first(status, "created_at", "createdAt", "published")
        favourited_text = _first(record, "favourited_at", "favorited_at", "created_at", "createdAt")
        tags = list(dict.fromkeys(["mastodon", "favourite", *_tags(status.get("tags"))]))
        if not any([status_id, url, content]):
            return None
        created_at = parse_datetime(created_text) or parse_datetime(favourited_text) or datetime.now(timezone.utc)
        updated_at = parse_datetime(favourited_text) or created_at
        reblog = status.get("reblog") if isinstance(status.get("reblog"), dict) else {}
        metadata = clean_metadata(
            {
                "status_id": status_id,
                "url": url,
                "external_url": url,
                "author": author,
                "acct": acct,
                "content": content,
                "created_at": created_text,
                "favourited_at": favourited_text,
                "language": _first(status, "language", "lang"),
                "reblog_status_id": _first(reblog, "id", "uri"),
                "reblog_url": _first(reblog, "url", "uri"),
                "reblog_author": _first(reblog.get("account", {}) if isinstance(reblog.get("account"), dict) else {}, "display_name", "acct", "username"),
                "reblogs_count": _number(status.get("reblogs_count")),
                "favourites_count": _number(status.get("favourites_count")),
                "replies_count": _number(status.get("replies_count")),
                "source_file": source_file,
            }
        )
        title = f"{author}: {content[:60]}" if author and content else (content[:80] or url or status_id)
        source_key = status_id or digest_source_id("status", url, content, source_file, index)
        return KnowledgeUnit(
            source_project="mastodon_favourites_json",
            source_id=f"mastodon_favourites_json:{source_key}" if status_id else digest_source_id("mastodon_favourites_json", source_key),
            source_entity_type="favourite",
            title=_html_text(title),
            content=_content(author, content, url, reblog),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at,
            updated_at=updated_at,
        )


def _iter_status_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("favourites", "favorites", "orderedItems", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [value]
    return []


def _status_record(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("status", "object", "item"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    return record


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


def _content(author: str, content: str, url: str, reblog: dict[str, Any]) -> str:
    parts = [f"Author: {author}" if author else "", content, f"URL: {url}" if url else ""]
    reblog_url = _first(reblog, "url", "uri")
    if reblog_url:
        parts.append(f"Reblog: {reblog_url}")
    return "\n".join(part for part in parts if part)
