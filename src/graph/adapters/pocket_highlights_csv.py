"""Adapter for Pocket highlight CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class PocketHighlightsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "pocket_highlights_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["highlight"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "highlight" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None
        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit and (sync_at is None or unit.updated_at > sync_at):
                    result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        text = first(row, "highlight", "highlight text", "text", "quote")
        if not text:
            return None
        url = first(row, "url", "article url", "item url", "resolved_url", "given_url")
        title = first(row, "title", "article title", "item title", "resolved_title", "given_title")
        note = first(row, "note", "notes", "annotation")
        created_at = parse_datetime(first(row, "created", "created_at", "created at", "time added", "date")) or datetime.now(timezone.utc)
        updated_at = parse_datetime(first(row, "updated", "updated_at", "updated at", "modified")) or created_at
        tags = [tag.casefold() for tag in split_values(first(row, "tags", "tag"))]
        metadata = clean_metadata({"url": url, "article_title": title, "highlight": text, "note": note, "tags": tags, "created_at": created_at.isoformat(), "updated_at": updated_at.isoformat(), "source_file": source_file})
        return KnowledgeUnit(source_project=self.name, source_id=digest_source_id(self.name, url, text), source_entity_type="highlight", title=title or text[:80], content=self._content(text, note, title, url), content_type=ContentType.ARTIFACT, metadata=metadata, tags=["pocket", *tags], created_at=created_at, updated_at=updated_at)

    def _content(self, text: str, note: str, title: str, url: str) -> str:
        return "\n".join(part for part in (text, f"Note: {note}" if note else "", f"Article: {title}" if title else "", f"URL: {url}" if url else "") if part)
