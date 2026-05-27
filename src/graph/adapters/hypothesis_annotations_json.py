"""Adapter for Hypothesis annotation JSON exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, iter_paths, parse_datetime, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class HypothesisAnnotationsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "hypothesis_annotations_json"

    @property
    def entity_types(self) -> list[str]:
        return ["annotation"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "annotation" not in entity_types:
            return result
        sync_at = since.last_sync_at if since else None
        if sync_at and sync_at.tzinfo is None:
            sync_at = sync_at.replace(tzinfo=timezone.utc)
        for path in iter_paths(self.path, {".json"}):
            for index, record in enumerate(_records(path)):
                unit = self._unit(record, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, record: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        text = str(record.get("text") or record.get("annotation") or "").strip()
        quote = str(record.get("quote") or record.get("exact") or record.get("selected_text") or "").strip()
        uri = str(record.get("uri") or record.get("url") or "").strip()
        if not any((text, quote, uri)):
            return None
        document = record.get("document") if isinstance(record.get("document"), dict) else {}
        title = _document_title(document) or uri or quote[:80] or "Hypothesis annotation"
        created_at = parse_datetime(record.get("created")) or datetime.now(timezone.utc)
        updated_at = parse_datetime(record.get("updated")) or created_at
        tags = [tag.casefold() for tag in split_values(record.get("tags"))]
        metadata = clean_metadata(
            {
                "annotation_id": record.get("id"),
                "text": text,
                "quote": quote,
                "uri": uri,
                "document_title": title,
                "tags": tags,
                "created": record.get("created"),
                "updated": record.get("updated"),
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project=SourceProject.HYPOTHESIS_ANNOTATIONS_JSON,
            source_id=digest_source_id("hypothesis_annotations_json", record.get("id") or uri, quote, text, index),
            source_entity_type="annotation",
            title=title,
            content="\n".join(part for part in (quote, text, uri) if part),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at,
            updated_at=updated_at,
        )


def _records(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("rows", "annotations", "items", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _document_title(document: dict[str, Any]) -> str:
    title = document.get("title")
    if isinstance(title, list):
        return str(title[0]).strip() if title else ""
    return str(title or "").strip()
