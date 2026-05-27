"""Adapter for Zotero Better BibTeX JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class ZoteroBetterBibtexJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "zotero_better_bibtex_json"

    @property
    def entity_types(self) -> list[str]:
        return ["bibliography_item"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "bibliography_item" not in set(entity_types or self.entity_types):
            return result
        sync_at = _ensure_utc(since.last_sync_at) if since else None
        for path in _iter_paths(self.path, ".json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for item in _items(data, "items"):
                unit = self._unit(item, path)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _unit(self, item: dict[str, Any], path: Path) -> KnowledgeUnit | None:
        key = _text(item.get("citationKey") or item.get("key") or item.get("itemKey"))
        title = _text(item.get("title"))
        abstract = _text(item.get("abstractNote") or item.get("abstract"))
        if not key and not title:
            return None
        creators = item.get("creators") or item.get("authors") or []
        tags = _strings(item.get("tags"))
        metadata = _clean(
            {
                "citation_key": key,
                "item_type": _text(item.get("itemType") or item.get("type")),
                "title": title,
                "abstract": abstract,
                "creators": creators,
                "doi": _text(item.get("DOI") or item.get("doi")),
                "url": _text(item.get("url")),
                "date": _text(item.get("date") or item.get("year")),
                "tags": tags,
                "collections": _strings(item.get("collections")),
                "source_file": path.name,
            }
        )
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=self.name,
            source_id=f"{self.name}:{_digest(key or title)}",
            source_entity_type="bibliography_item",
            title=title or key,
            content=abstract or _citation_body(title, creators, metadata.get("date"), metadata.get("doi")),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["zotero", *tags],
            created_at=now,
            updated_at=now,
        )


def _citation_body(title: str, creators: Any, date: Any, doi: Any) -> str:
    authors = ", ".join(_strings(creators))
    return "\n".join(part for part in (title, authors, str(date or ""), f"DOI: {doi}" if doi else "") if part)


def _items(data: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        nested = data.get(key) or data.get("data") or data.get("results")
        if nested is not None:
            return _items(nested, key)
        return [data]
    return []


def _iter_paths(path: str, suffix: str) -> list[Path]:
    root = Path(path).expanduser()
    if root.is_file() and root.suffix.lower() == suffix:
        return [root]
    return sorted(root.rglob(f"*{suffix}")) if root.is_dir() else []


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = [part.strip() for part in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        raw = [_text(item.get("name") or item.get("tag") or item.get("lastName") or item.get("title") or item) if isinstance(item, dict) else _text(item) for item in value]
    else:
        raw = []
    return [item for item in dict.fromkeys(raw) if item]


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _clean(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if value not in ("", None, [])}


def _digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()[:24]


def _ensure_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
