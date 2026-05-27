"""Adapter for Omnivore highlight JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, iter_paths, parse_datetime, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class OmnivoreHighlightsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "omnivore_highlights_json"

    @property
    def entity_types(self) -> list[str]:
        return ["highlight"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "highlight" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        if sync_at and sync_at.tzinfo is None:
            sync_at = sync_at.replace(tzinfo=timezone.utc)
        for path in iter_paths(self.path, {".json"}):
            for index, item in enumerate(_highlight_records(path)):
                unit = self._unit(item, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, item: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        article = item.get("article") if isinstance(item.get("article"), dict) else {}
        text = str(item.get("text") or item.get("highlight") or item.get("quote") or "").strip()
        note = str(item.get("note") or item.get("annotation") or "").strip()
        title = str(item.get("title") or article.get("title") or "").strip()
        url = str(item.get("url") or article.get("url") or article.get("originalUrl") or "").strip()
        if not any((text, note, title, url)):
            return None
        created_at = parse_datetime(item.get("highlightedAt") or item.get("createdAt") or item.get("created")) or datetime.now(timezone.utc)
        labels = _labels(item.get("labels") or article.get("labels") or item.get("tags"))
        metadata = clean_metadata(
            {
                "highlight_id": item.get("id"),
                "article_id": article.get("id") or item.get("article_id"),
                "article_title": title,
                "url": url,
                "author": item.get("author") or article.get("author"),
                "highlight_text": text,
                "note": note,
                "labels": labels,
                "highlighted_at": created_at.isoformat(),
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.OMNIVORE_HIGHLIGHTS_JSON,
            source_id=digest_source_id("omnivore_highlights_json", item.get("id") or article.get("id") or url, text, index),
            source_entity_type="highlight",
            title=title or text[:80] or "Omnivore highlight",
            content="\n".join(part for part in (title, url, text, note) if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=labels,
            created_at=created_at,
            updated_at=created_at,
        )


def _highlight_records(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    records: list[dict[str, Any]] = []
    roots = data if isinstance(data, list) else [data]
    for root in roots:
        if not isinstance(root, dict):
            continue
        highlights = root.get("highlights")
        if isinstance(highlights, list):
            for highlight in highlights:
                if isinstance(highlight, dict):
                    merged = {**highlight, "article": root}
                    records.append(merged)
        elif any(key in root for key in ("text", "highlight", "quote")):
            records.append(root)
        for key in ("items", "articles", "data"):
            value = root.get(key)
            if isinstance(value, list):
                for child in value:
                    if isinstance(child, dict):
                        records.extend(_records_from_object(child))
    return records


def _records_from_object(root: dict[str, Any]) -> list[dict[str, Any]]:
    highlights = root.get("highlights")
    if isinstance(highlights, list):
        return [{**highlight, "article": root} for highlight in highlights if isinstance(highlight, dict)]
    return [root]


def _labels(value: object) -> list[str]:
    if isinstance(value, list):
        raw = [item.get("name") if isinstance(item, dict) else item for item in value]
    else:
        raw = split_values(value)
    labels: list[str] = []
    for item in raw:
        text = str(item or "").strip().casefold()
        if text and text not in labels:
            labels.append(text)
    return labels
